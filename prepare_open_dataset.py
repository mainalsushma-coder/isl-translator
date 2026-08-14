import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# =========================================================
# Paths and configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

HAND_MODEL_PATH = (
    PROJECT_ROOT / "models" / "hand_landmarker.task"
)

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "external_data"
    / "A Video Dataset of the Hand Gestures of Indian Sign Language Words used in Emergency Situations"
    / "Raw_Data"
)

OUTPUT_PATH = PROJECT_ROOT / "data" / "sequences_v3"
MANIFEST_PATH = OUTPUT_PATH / "manifest.csv"

SEQUENCE_LENGTH = 48

# Per frame:
# Left hand:  63 local landmarks + wrist x/y = 65
# Right hand: 63 local landmarks + wrist x/y = 65
# Presence:   left + right = 2
FEATURE_COUNT = 132

MIN_DETECTION_RATIO = 0.30

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


# =========================================================
# Normalize one hand
# =========================================================

def normalize_hand(landmarks):
    """
    Produce 65 values for one hand:

    - 63 wrist-relative and scale-normalized landmark values
    - wrist x and y coordinates for hand movement tracking

    Wrist coordinates are retained because removing them would
    also remove the movement path of a dynamic sign.
    """

    wrist = landmarks[0]

    relative_points = []

    for landmark in landmarks:
        relative_points.append(
            (
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z,
            )
        )

    scale = max(
        max(abs(x), abs(y), abs(z))
        for x, y, z in relative_points
    )

    if scale == 0:
        scale = 1.0

    features = []

    for x, y, z in relative_points:
        features.extend(
            [
                x / scale,
                y / scale,
                z / scale,
            ]
        )

    # Preserve the hand's position inside the frame.
    features.extend([wrist.x, wrist.y])

    return np.asarray(features, dtype=np.float32)


# =========================================================
# Extract 132 features from one frame
# =========================================================

def extract_frame_features(result):
    left_features = np.zeros(65, dtype=np.float32)
    right_features = np.zeros(65, dtype=np.float32)

    left_present = 0.0
    right_present = 0.0

    left_score = -1.0
    right_score = -1.0

    if result.hand_landmarks:
        for index, landmarks in enumerate(result.hand_landmarks):

            hand_label = ""
            confidence = 0.0

            if index < len(result.handedness):
                categories = result.handedness[index]

                if categories:
                    hand_label = (
                        categories[0].category_name or ""
                    ).upper()

                    confidence = float(
                        categories[0].score or 0.0
                    )

            hand_features = normalize_hand(landmarks)

            if hand_label == "LEFT":
                if confidence > left_score:
                    left_features = hand_features
                    left_present = 1.0
                    left_score = confidence

            elif hand_label == "RIGHT":
                if confidence > right_score:
                    right_features = hand_features
                    right_present = 1.0
                    right_score = confidence

            else:
                # Fallback if MediaPipe does not return handedness.
                wrist_x = landmarks[0].x

                if wrist_x < 0.5:
                    left_features = hand_features
                    left_present = 1.0
                else:
                    right_features = hand_features
                    right_present = 1.0

    features = np.concatenate(
        [
            left_features,
            right_features,
            np.asarray(
                [left_present, right_present],
                dtype=np.float32,
            ),
        ]
    )

    hand_detected = bool(left_present or right_present)

    return features, hand_detected


# =========================================================
# Process one OpenCV frame
# =========================================================

def process_frame(frame, landmarker):
    # Match our selfie-style live webcam pipeline.
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    rgb_frame = np.ascontiguousarray(rgb_frame)

    mediapipe_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    result = landmarker.detect(mediapipe_image)

    return extract_frame_features(result)


# =========================================================
# Convert one video into exactly 48 frames
# =========================================================

def extract_video_sequence(video_path, landmarker):
    camera = cv2.VideoCapture(str(video_path))

    if not camera.isOpened():
        raise RuntimeError("OpenCV could not open the video.")

    total_frames = int(
        camera.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    sequence = []
    detection_values = []

    if total_frames > 0:
        target_indices = np.rint(
            np.linspace(
                0,
                total_frames - 1,
                SEQUENCE_LENGTH,
            )
        ).astype(int)

        target_position = 0
        frame_index = 0

        while target_position < SEQUENCE_LENGTH:
            success, frame = camera.read()

            if not success:
                break

            if frame_index == target_indices[target_position]:
                features, detected = process_frame(
                    frame,
                    landmarker,
                )

                # Some short videos may require one frame
                # to appear more than once in the sequence.
                while (
                    target_position < SEQUENCE_LENGTH
                    and target_indices[target_position]
                    == frame_index
                ):
                    sequence.append(features.copy())
                    detection_values.append(float(detected))
                    target_position += 1

            frame_index += 1

    else:
        # Fallback for a video whose frame count metadata
        # cannot be read correctly.
        all_features = []
        all_detections = []

        while True:
            success, frame = camera.read()

            if not success:
                break

            features, detected = process_frame(
                frame,
                landmarker,
            )

            all_features.append(features)
            all_detections.append(float(detected))

        if all_features:
            selected_indices = np.rint(
                np.linspace(
                    0,
                    len(all_features) - 1,
                    SEQUENCE_LENGTH,
                )
            ).astype(int)

            sequence = [
                all_features[index]
                for index in selected_indices
            ]

            detection_values = [
                all_detections[index]
                for index in selected_indices
            ]

    camera.release()

    if not sequence:
        raise RuntimeError("No readable frames were found.")

    # Pad if OpenCV stopped before its reported frame count.
    while len(sequence) < SEQUENCE_LENGTH:
        sequence.append(sequence[-1].copy())
        detection_values.append(detection_values[-1])

    sequence = np.asarray(
        sequence[:SEQUENCE_LENGTH],
        dtype=np.float32,
    )

    detection_ratio = float(
        np.mean(detection_values[:SEQUENCE_LENGTH])
    )

    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURE_COUNT,
    ):
        raise ValueError(
            f"Unexpected sequence shape: {sequence.shape}"
        )

    return sequence, detection_ratio


# =========================================================
# Read signer and recording number from filename
# =========================================================

def parse_video_identity(video_path):
    # Example filename:
    # doctor_001_02.avi

    parts = video_path.stem.split("_")

    if len(parts) >= 3:
        signer_id = parts[-2]
        recording_id = parts[-1]
    else:
        signer_id = "UNKNOWN"
        recording_id = video_path.stem

    return signer_id, recording_id


# =========================================================
# Write a CSV describing all generated sequences
# =========================================================

def write_manifest():
    rows = []

    for sequence_path in sorted(
        OUTPUT_PATH.rglob("*.npz")
    ):
        with np.load(
            sequence_path,
            allow_pickle=False,
        ) as data:
            sequence = data["sequence"]

            rows.append(
                {
                    "sequence_path": str(
                        sequence_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "label": str(data["label"].item()),
                    "signer_id": str(
                        data["signer_id"].item()
                    ),
                    "recording_id": str(
                        data["recording_id"].item()
                    ),
                    "source": str(data["source"].item()),
                    "original_video": str(
                        data["original_video"].item()
                    ),
                    "detection_ratio": float(
                        data["detection_ratio"].item()
                    ),
                    "frames": sequence.shape[0],
                    "features": sequence.shape[1],
                }
            )

    if not rows:
        return

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# Main dataset processing
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert external ISL videos into "
            "MediaPipe landmark sequences."
        )
    )

    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional labels to process, such as DOCTOR HELP.",
    )

    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum videos to process per label.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace sequences that already exist.",
    )

    arguments = parser.parse_args()

    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Hand model not found:\n{HAND_MODEL_PATH}"
        )

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_DATA_PATH}"
        )

    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    requested_labels = None

    if arguments.labels:
        requested_labels = {
            label.upper()
            for label in arguments.labels
        }

    label_folders = sorted(
        folder
        for folder in RAW_DATA_PATH.iterdir()
        if folder.is_dir()
    )

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(HAND_MODEL_PATH)
        ),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    saved_count = 0
    skipped_count = 0
    failed_count = 0

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        for label_folder in label_folders:
            label = (
                label_folder.name
                .removesuffix("_Raw")
                .upper()
            )

            if (
                requested_labels is not None
                and label not in requested_labels
            ):
                continue

            video_paths = sorted(
                path
                for path in label_folder.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in VIDEO_EXTENSIONS
                )
            )

            if arguments.max_videos is not None:
                video_paths = video_paths[
                    :arguments.max_videos
                ]

            label_output_path = OUTPUT_PATH / label

            label_output_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            print(
                f"\n{label}: "
                f"{len(video_paths)} video(s)"
            )

            for video_number, video_path in enumerate(
                video_paths,
                start=1,
            ):
                signer_id, recording_id = (
                    parse_video_identity(video_path)
                )

                output_file = (
                    label_output_path
                    / (
                        f"{label}_"
                        f"{signer_id}_"
                        f"{recording_id}.npz"
                    )
                )

                if (
                    output_file.exists()
                    and not arguments.overwrite
                ):
                    print(
                        f"  [{video_number}/"
                        f"{len(video_paths)}] SKIPPED: "
                        f"{output_file.name}"
                    )

                    skipped_count += 1
                    continue

                try:
                    sequence, detection_ratio = (
                        extract_video_sequence(
                            video_path,
                            landmarker,
                        )
                    )

                    if (
                        detection_ratio
                        < MIN_DETECTION_RATIO
                    ):
                        print(
                            f"  [{video_number}/"
                            f"{len(video_paths)}] FAILED: "
                            f"{video_path.name} — "
                            f"hand detection only "
                            f"{detection_ratio * 100:.1f}%"
                        )

                        failed_count += 1
                        continue

                    np.savez_compressed(
                        output_file,
                        sequence=sequence,
                        label=np.asarray(label),
                        signer_id=np.asarray(
                            f"MENDELEY_{signer_id}"
                        ),
                        recording_id=np.asarray(
                            recording_id
                        ),
                        source=np.asarray(
                            "MENDELEY_EMERGENCY_ISL"
                        ),
                        original_video=np.asarray(
                            str(video_path)
                        ),
                        detection_ratio=np.asarray(
                            detection_ratio,
                            dtype=np.float32,
                        ),
                    )

                    print(
                        f"  [{video_number}/"
                        f"{len(video_paths)}] SAVED: "
                        f"{output_file.name} — "
                        f"{sequence.shape} — "
                        f"detection "
                        f"{detection_ratio * 100:.1f}%"
                    )

                    saved_count += 1

                except Exception as error:
                    print(
                        f"  [{video_number}/"
                        f"{len(video_paths)}] ERROR: "
                        f"{video_path.name} — {error}"
                    )

                    failed_count += 1

    write_manifest()

    print("\nDataset preparation complete")
    print("----------------------------")
    print(f"Saved: {saved_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Features per frame: {FEATURE_COUNT}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()