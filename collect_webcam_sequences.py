import argparse
import re
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from prepare_open_dataset import (
    extract_frame_features,
    write_manifest,
)


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

HAND_MODEL_PATH = (
    PROJECT_ROOT / "models" / "hand_landmarker.task"
)

SEQUENCE_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "sequences_v3"
)

VIDEO_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "webcam_videos_v3"
)

SEQUENCE_LENGTH = 48
FEATURE_COUNT = 132

COUNTDOWN_SECONDS = 2.0
MIN_DETECTION_RATIO = 0.70


# =========================================================
# Save the corresponding webcam video
# =========================================================

def save_video(frames, output_path, fps):
    if not frames:
        return False

    height, width, _ = frames[0].shape

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        return False

    for frame in frames:
        writer.write(frame)

    writer.release()

    return True


# =========================================================
# Main collector
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect webcam adaptation sequences "
            "for the ISL V3 model."
        )
    )

    parser.add_argument(
        "--label",
        required=True,
        help="Gesture label, for example PAIN.",
    )

    parser.add_argument(
        "--signer",
        required=True,
        help="Signer ID, for example SUSHMA.",
    )

    parser.add_argument(
        "--clips",
        type=int,
        default=5,
        help="Number of clips to collect.",
    )

    arguments = parser.parse_args()

    label = re.sub(
        r"[^A-Z0-9_]+",
        "_",
        arguments.label.strip().upper(),
    )

    signer = re.sub(
        r"[^A-Z0-9_]+",
        "_",
        arguments.signer.strip().upper(),
    )

    signer_id = f"WEBCAM_{signer}"

    if not label:
        raise ValueError("Label cannot be empty.")

    if not signer:
        raise ValueError("Signer cannot be empty.")

    if arguments.clips <= 0:
        raise ValueError(
            "Number of clips must be greater than zero."
        )

    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Hand model not found:\n"
            f"{HAND_MODEL_PATH}"
        )

    sequence_label_path = (
        SEQUENCE_OUTPUT_PATH / label
    )

    video_label_path = VIDEO_OUTPUT_PATH / label

    sequence_label_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_label_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # MediaPipe setup
    # -----------------------------------------------------

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
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    camera_fps = camera.get(cv2.CAP_PROP_FPS)

    if camera_fps <= 0 or camera_fps > 120:
        camera_fps = 30.0

    state = "READY"
    countdown_end_time = 0.0

    captured_features = []
    captured_frames = []
    detected_frames = 0

    saved_clips = 0
    last_detection_ratio = 0.0
    last_timestamp_ms = 0

    print("Webcam Sequence Collector V3")
    print("----------------------------")
    print(f"Label: {label}")
    print(f"Signer: {signer_id}")
    print(f"Target clips: {arguments.clips}")
    print("Press R to record each clip.")
    print("Press Q to quit.")

    try:
        with HandLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = camera.read()

                if not success:
                    break

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                rgb_frame = np.ascontiguousarray(
                    rgb_frame
                )

                mediapipe_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                current_timestamp_ms = (
                    time.perf_counter_ns()
                    // 1_000_000
                )

                timestamp_ms = max(
                    current_timestamp_ms,
                    last_timestamp_ms + 1,
                )

                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    mediapipe_image,
                    timestamp_ms,
                )

                features, hand_detected = (
                    extract_frame_features(result)
                )

                if len(features) != FEATURE_COUNT:
                    raise ValueError(
                        f"Expected {FEATURE_COUNT} "
                        f"features, received "
                        f"{len(features)}."
                    )

                frame_height, frame_width, _ = (
                    frame.shape
                )

                if result.hand_landmarks:
                    for hand_landmarks in (
                        result.hand_landmarks
                    ):
                        for landmark in hand_landmarks:
                            x = int(
                                landmark.x
                                * frame_width
                            )
                            y = int(
                                landmark.y
                                * frame_height
                            )

                            cv2.circle(
                                frame,
                                (x, y),
                                4,
                                (0, 255, 0),
                                -1,
                            )

                current_time = time.perf_counter()

                # -----------------------------------------
                # Countdown
                # -----------------------------------------

                if state == "COUNTDOWN":
                    if (
                        current_time
                        >= countdown_end_time
                    ):
                        state = "RECORDING"
                        captured_features = []
                        captured_frames = []
                        detected_frames = 0

                # -----------------------------------------
                # Record 48 frames
                # -----------------------------------------

                if state == "RECORDING":
                    captured_features.append(
                        features.copy()
                    )

                    captured_frames.append(
                        frame.copy()
                    )

                    if hand_detected:
                        detected_frames += 1

                    if (
                        len(captured_features)
                        >= SEQUENCE_LENGTH
                    ):
                        sequence = np.asarray(
                            captured_features[
                                :SEQUENCE_LENGTH
                            ],
                            dtype=np.float32,
                        )

                        last_detection_ratio = (
                            detected_frames
                            / SEQUENCE_LENGTH
                        )

                        if (
                            last_detection_ratio
                            < MIN_DETECTION_RATIO
                        ):
                            state = "REJECTED"

                            print(
                                "Clip rejected — "
                                f"hand detection "
                                f"{last_detection_ratio * 100:.1f}%"
                            )

                        else:
                            timestamp_text = (
                                time.strftime(
                                    "%Y%m%d_%H%M%S"
                                )
                            )

                            recording_id = (
                                f"{timestamp_text}_"
                                f"{saved_clips + 1:02d}"
                            )

                            file_stem = (
                                f"{label}_"
                                f"{signer_id}_"
                                f"{recording_id}"
                            )

                            sequence_path = (
                                sequence_label_path
                                / f"{file_stem}.npz"
                            )

                            video_path = (
                                video_label_path
                                / f"{file_stem}.avi"
                            )

                            video_saved = save_video(
                                captured_frames,
                                video_path,
                                camera_fps,
                            )

                            original_video_value = (
                                str(video_path)
                                if video_saved
                                else "VIDEO_NOT_SAVED"
                            )

                            np.savez_compressed(
                                sequence_path,
                                sequence=sequence,
                                label=np.asarray(label),
                                signer_id=np.asarray(
                                    signer_id
                                ),
                                recording_id=np.asarray(
                                    recording_id
                                ),
                                source=np.asarray(
                                    "WEBCAM_ADAPTATION"
                                ),
                                original_video=np.asarray(
                                    original_video_value
                                ),
                                detection_ratio=np.asarray(
                                    last_detection_ratio,
                                    dtype=np.float32,
                                ),
                            )

                            saved_clips += 1

                            write_manifest()

                            print(
                                f"Saved clip "
                                f"{saved_clips}/"
                                f"{arguments.clips}: "
                                f"{sequence_path.name} — "
                                f"detection "
                                f"{last_detection_ratio * 100:.1f}%"
                            )

                            if (
                                saved_clips
                                >= arguments.clips
                            ):
                                state = "COMPLETE"
                            else:
                                state = "SAVED"

                # =========================================
                # Interface
                # =========================================

                cv2.rectangle(
                    frame,
                    (0, 0),
                    (frame_width, 190),
                    (20, 20, 20),
                    -1,
                )

                cv2.putText(
                    frame,
                    f"Label: {label}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Saved: {saved_clips}/"
                        f"{arguments.clips}"
                    ),
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )

                if state == "COUNTDOWN":
                    remaining = max(
                        0,
                        int(
                            countdown_end_time
                            - current_time
                        )
                        + 1,
                    )

                    instruction = (
                        f"Starting in {remaining}"
                    )
                    colour = (0, 255, 255)

                elif state == "RECORDING":
                    instruction = (
                        "RECORDING: "
                        f"{len(captured_features)}/"
                        f"{SEQUENCE_LENGTH}"
                    )
                    colour = (0, 0, 255)

                elif state == "REJECTED":
                    instruction = (
                        "Rejected: keep hand visible. "
                        "Press R again"
                    )
                    colour = (0, 0, 255)

                elif state == "SAVED":
                    instruction = (
                        "Saved! Press R for next clip"
                    )
                    colour = (0, 255, 0)

                elif state == "COMPLETE":
                    instruction = (
                        "Collection complete. Press Q"
                    )
                    colour = (0, 255, 0)

                else:
                    instruction = (
                        "Press R to record"
                    )
                    colour = (0, 255, 255)

                cv2.putText(
                    frame,
                    instruction,
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    colour,
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        "Start neutral, perform sign "
                        "once, finish the movement"
                    ),
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    "R: record | Q: quit",
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(
                    "Webcam Adaptation Collector V3",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if (
                    key == ord("r")
                    and state != "RECORDING"
                    and saved_clips
                    < arguments.clips
                ):
                    state = "COUNTDOWN"

                    countdown_end_time = (
                        time.perf_counter()
                        + COUNTDOWN_SECONDS
                    )

                    captured_features = []
                    captured_frames = []
                    detected_frames = 0

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("\nCollection finished")
    print("-------------------")
    print(f"Label: {label}")
    print(f"Saved clips: {saved_clips}")
    print(f"Sequences: {sequence_label_path}")
    print(f"Videos: {video_label_path}")


if __name__ == "__main__":
    main()