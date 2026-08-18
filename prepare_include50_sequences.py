"""Prepare the clean AI4Bharat INCLUDE-50 videos for model training.

This script reads the manifest produced by ``download_include50.py`` and
converts every downloaded video into the same landmark representation used by
the emergency-sign V3 pipeline:

* 48 frames per video
* 132 float32 features per frame
* left hand: 63 wrist-relative landmarks + wrist x/y
* right hand: 63 wrist-relative landmarks + wrist x/y
* two hand-presence flags

The official INCLUDE train/validation/test split is preserved in both the
output directory layout and the generated processing manifest. INCLUDE's
Hugging Face metadata does not provide reliable signer identities, so this
script deliberately records the signer as unavailable rather than inventing
one.

The run is resumable. Existing valid sequence files are skipped unless
``--overwrite`` is supplied, and new files are written atomically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path, PurePosixPath

import cv2
import mediapipe as mp
import numpy as np


# =========================================================
# Paths and configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

HAND_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

DOWNLOAD_ROOT = PROJECT_ROOT / "external_data" / "INCLUDE50"
DOWNLOAD_MANIFEST_PATH = DOWNLOAD_ROOT / "include50_manifest.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "sequences_include50_v4"
PROCESSING_MANIFEST_PATH = OUTPUT_PATH / "manifest.csv"

SEQUENCE_LENGTH = 48

# Per frame:
# Left hand:  63 local landmarks + wrist x/y = 65
# Right hand: 63 local landmarks + wrist x/y = 65
# Presence:   left + right = 2
FEATURE_COUNT = 132

DEFAULT_MIN_DETECTION_RATIO = 0.30
EXPECTED_SPLITS = {"train", "val", "test"}

PROCESSING_MANIFEST_FIELDS = [
    "video_path",
    "original_video",
    "sequence_path",
    "split",
    "source_label",
    "label",
    "signer_id",
    "recording_id",
    "source",
    "detection_ratio",
    "left_hand_ratio",
    "right_hand_ratio",
    "two_hand_ratio",
    "frames",
    "features",
    "status",
    "error",
]


# =========================================================
# Command-line arguments
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert downloaded AI4Bharat INCLUDE-50 videos into "
            "48 x 132 MediaPipe landmark sequences."
        )
    )

    parser.add_argument(
        "--labels",
        nargs="*",
        help=(
            "Optional normalized labels to process, for example "
            "HELLO THANK_YOU GOOD_MORNING."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        choices=sorted(EXPECTED_SPLITS),
        help="Optional official splits to process: train val test.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum total videos to process after applying filters.",
    )
    parser.add_argument(
        "--min-detection-ratio",
        type=float,
        default=DEFAULT_MIN_DETECTION_RATIO,
        help=(
            "Minimum fraction of sampled frames containing at least one "
            f"detected hand (default: {DEFAULT_MIN_DETECTION_RATIO:.2f})."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess valid sequence files that already exist.",
    )

    args = parser.parse_args()

    if args.max_videos is not None and args.max_videos <= 0:
        parser.error("--max-videos must be greater than zero.")

    if not 0.0 <= args.min_detection_ratio <= 1.0:
        parser.error("--min-detection-ratio must be between 0 and 1.")

    return args


# =========================================================
# Labels, paths and manifests
# =========================================================

def normalize_model_label(value: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", value.strip().upper())
    return normalized.strip("_")


def load_download_manifest() -> list[dict[str, str]]:
    if not DOWNLOAD_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "INCLUDE-50 download manifest was not found:\n"
            f"{DOWNLOAD_MANIFEST_PATH}"
        )

    with DOWNLOAD_MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("The INCLUDE-50 download manifest is empty.")

    required_fields = {
        "split",
        "source_label",
        "model_label",
        "video_path",
        "local_path",
        "status",
    }
    missing_fields = required_fields.difference(rows[0])

    if missing_fields:
        raise ValueError(
            "Download manifest is missing required columns: "
            + ", ".join(sorted(missing_fields))
        )

    seen_paths: set[str] = set()

    for row in rows:
        video_path = row["video_path"].replace("\\", "/")

        if video_path in seen_paths:
            raise ValueError(
                f"Duplicate video_path in download manifest: {video_path}"
            )

        seen_paths.add(video_path)
        row["video_path"] = video_path
        row["split"] = row["split"].strip().lower()
        row["model_label"] = normalize_model_label(row["model_label"])

        if row["split"] not in EXPECTED_SPLITS:
            raise ValueError(
                f"Unexpected split {row['split']!r} for {video_path}"
            )

        if not row["model_label"]:
            raise ValueError(f"Empty model label for {video_path}")

    return sorted(
        rows,
        key=lambda row: (
            row["split"],
            row["model_label"],
            row["video_path"].casefold(),
        ),
    )


def load_previous_processing_rows() -> dict[str, dict[str, str]]:
    if not PROCESSING_MANIFEST_PATH.exists():
        return {}

    with PROCESSING_MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        rows = list(csv.DictReader(file))

    return {
        row["video_path"].replace("\\", "/"): row
        for row in rows
        if row.get("video_path")
    }


def resolve_video_path(row: dict[str, str]) -> Path:
    local_path = Path(row["local_path"])

    if not local_path.is_absolute():
        local_path = PROJECT_ROOT / local_path

    return local_path.resolve()


def output_path_for(row: dict[str, str]) -> Path:
    video_identity = row["video_path"].encode("utf-8")
    short_hash = hashlib.sha1(video_identity).hexdigest()[:10]
    video_stem = normalize_model_label(
        PurePosixPath(row["video_path"]).stem
    )

    filename = f"{row['model_label']}_{video_stem}_{short_hash}.npz"

    return (
        OUTPUT_PATH
        / row["split"]
        / row["model_label"]
        / filename
    )


def relative_project_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def blank_processing_row(
    source_row: dict[str, str],
) -> dict[str, str]:
    video_path = resolve_video_path(source_row)
    sequence_path = output_path_for(source_row)

    return {
        "video_path": source_row["video_path"],
        "original_video": relative_project_path(video_path),
        "sequence_path": relative_project_path(sequence_path),
        "split": source_row["split"],
        "source_label": source_row["source_label"],
        "label": source_row["model_label"],
        "signer_id": "UNAVAILABLE_IN_INCLUDE_METADATA",
        "recording_id": PurePosixPath(source_row["video_path"]).stem,
        "source": "AI4BHARAT_INCLUDE50",
        "detection_ratio": "",
        "left_hand_ratio": "",
        "right_hand_ratio": "",
        "two_hand_ratio": "",
        "frames": "",
        "features": "",
        "status": "PENDING",
        "error": "",
    }


def write_processing_manifest(
    processing_rows: dict[str, dict[str, str]],
) -> None:
    PROCESSING_MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = PROCESSING_MANIFEST_PATH.with_suffix(
        ".csv.partial"
    )

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=PROCESSING_MANIFEST_FIELDS,
        )
        writer.writeheader()

        for video_path in sorted(
            processing_rows,
            key=str.casefold,
        ):
            row = processing_rows[video_path]
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in PROCESSING_MANIFEST_FIELDS
                }
            )

    temporary_path.replace(PROCESSING_MANIFEST_PATH)


# =========================================================
# Normalize one hand
# =========================================================

def normalize_hand(landmarks) -> np.ndarray:
    """Produce 65 values using the emergency V3 representation."""

    wrist = landmarks[0]

    relative_points = [
        (
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z,
        )
        for landmark in landmarks
    ]

    scale = max(
        max(abs(x), abs(y), abs(z))
        for x, y, z in relative_points
    )

    if scale == 0:
        scale = 1.0

    features: list[float] = []

    for x, y, z in relative_points:
        features.extend([x / scale, y / scale, z / scale])

    # Retain the hand's frame position for dynamic movement tracking.
    features.extend([wrist.x, wrist.y])

    return np.asarray(features, dtype=np.float32)


# =========================================================
# Extract 132 features from one frame
# =========================================================

def extract_frame_features(result) -> tuple[np.ndarray, float, float]:
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
                    confidence = float(categories[0].score or 0.0)

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

    return features, left_present, right_present


# =========================================================
# Process one OpenCV frame
# =========================================================

def process_frame(
    frame: np.ndarray,
    landmarker,
) -> tuple[np.ndarray, float, float]:
    # Match the selfie-style emergency and live-webcam pipelines.
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

def extract_video_sequence(
    video_path: Path,
    landmarker,
) -> tuple[np.ndarray, dict[str, float]]:
    camera = cv2.VideoCapture(str(video_path))

    if not camera.isOpened():
        raise RuntimeError("OpenCV could not open the video.")

    total_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

    sequence: list[np.ndarray] = []
    left_values: list[float] = []
    right_values: list[float] = []

    try:
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
                    features, left_present, right_present = (
                        process_frame(frame, landmarker)
                    )

                    # Short videos may repeat a sampled frame.
                    while (
                        target_position < SEQUENCE_LENGTH
                        and target_indices[target_position] == frame_index
                    ):
                        sequence.append(features.copy())
                        left_values.append(left_present)
                        right_values.append(right_present)
                        target_position += 1

                frame_index += 1

        else:
            # Fallback when frame-count metadata cannot be read.
            all_features: list[np.ndarray] = []
            all_left_values: list[float] = []
            all_right_values: list[float] = []

            while True:
                success, frame = camera.read()

                if not success:
                    break

                features, left_present, right_present = process_frame(
                    frame,
                    landmarker,
                )

                all_features.append(features)
                all_left_values.append(left_present)
                all_right_values.append(right_present)

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
                left_values = [
                    all_left_values[index]
                    for index in selected_indices
                ]
                right_values = [
                    all_right_values[index]
                    for index in selected_indices
                ]

    finally:
        camera.release()

    if not sequence:
        raise RuntimeError("No readable frames were found.")

    # Pad if OpenCV stops before its reported frame count.
    while len(sequence) < SEQUENCE_LENGTH:
        sequence.append(sequence[-1].copy())
        left_values.append(left_values[-1])
        right_values.append(right_values[-1])

    sequence_array = np.asarray(
        sequence[:SEQUENCE_LENGTH],
        dtype=np.float32,
    )
    left_array = np.asarray(
        left_values[:SEQUENCE_LENGTH],
        dtype=np.float32,
    )
    right_array = np.asarray(
        right_values[:SEQUENCE_LENGTH],
        dtype=np.float32,
    )

    if sequence_array.shape != (SEQUENCE_LENGTH, FEATURE_COUNT):
        raise ValueError(
            f"Unexpected sequence shape: {sequence_array.shape}"
        )

    any_hand = np.maximum(left_array, right_array)
    both_hands = np.minimum(left_array, right_array)

    ratios = {
        "detection_ratio": float(np.mean(any_hand)),
        "left_hand_ratio": float(np.mean(left_array)),
        "right_hand_ratio": float(np.mean(right_array)),
        "two_hand_ratio": float(np.mean(both_hands)),
    }

    return sequence_array, ratios


# =========================================================
# Sequence validation and saving
# =========================================================

def valid_existing_sequence(
    output_file: Path,
    source_row: dict[str, str],
) -> bool:
    if not output_file.exists():
        return False

    try:
        with np.load(output_file, allow_pickle=False) as data:
            return (
                data["sequence"].shape
                == (SEQUENCE_LENGTH, FEATURE_COUNT)
                and str(data["label"].item())
                == source_row["model_label"]
                and str(data["split"].item()) == source_row["split"]
                and str(data["video_path"].item())
                == source_row["video_path"]
            )
    except Exception:
        return False


def metadata_from_existing_sequence(
    output_file: Path,
) -> dict[str, str]:
    with np.load(output_file, allow_pickle=False) as data:
        return {
            "detection_ratio": str(
                float(data["detection_ratio"].item())
            ),
            "left_hand_ratio": str(
                float(data["left_hand_ratio"].item())
            ),
            "right_hand_ratio": str(
                float(data["right_hand_ratio"].item())
            ),
            "two_hand_ratio": str(
                float(data["two_hand_ratio"].item())
            ),
            "frames": str(data["sequence"].shape[0]),
            "features": str(data["sequence"].shape[1]),
            "status": "READY",
            "error": "",
        }


def save_sequence_atomically(
    output_file: Path,
    sequence: np.ndarray,
    ratios: dict[str, float],
    source_row: dict[str, str],
    original_video: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_file.with_suffix(".npz.partial")

    with temporary_path.open("wb") as file:
        np.savez_compressed(
            file,
            sequence=sequence,
            label=np.asarray(source_row["model_label"]),
            source_label=np.asarray(source_row["source_label"]),
            split=np.asarray(source_row["split"]),
            signer_id=np.asarray(
                "UNAVAILABLE_IN_INCLUDE_METADATA"
            ),
            recording_id=np.asarray(
                PurePosixPath(source_row["video_path"]).stem
            ),
            source=np.asarray("AI4BHARAT_INCLUDE50"),
            original_video=np.asarray(str(original_video)),
            video_path=np.asarray(source_row["video_path"]),
            detection_ratio=np.asarray(
                ratios["detection_ratio"],
                dtype=np.float32,
            ),
            left_hand_ratio=np.asarray(
                ratios["left_hand_ratio"],
                dtype=np.float32,
            ),
            right_hand_ratio=np.asarray(
                ratios["right_hand_ratio"],
                dtype=np.float32,
            ),
            two_hand_ratio=np.asarray(
                ratios["two_hand_ratio"],
                dtype=np.float32,
            ),
        )

    temporary_path.replace(output_file)


def update_ready_row(
    processing_row: dict[str, str],
    ratios: dict[str, float],
) -> None:
    processing_row.update(
        {
            "detection_ratio": str(ratios["detection_ratio"]),
            "left_hand_ratio": str(ratios["left_hand_ratio"]),
            "right_hand_ratio": str(ratios["right_hand_ratio"]),
            "two_hand_ratio": str(ratios["two_hand_ratio"]),
            "frames": str(SEQUENCE_LENGTH),
            "features": str(FEATURE_COUNT),
            "status": "READY",
            "error": "",
        }
    )


# =========================================================
# Main processing loop
# =========================================================

def main() -> int:
    args = parse_args()

    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Hand model not found:\n{HAND_MODEL_PATH}"
        )

    source_rows = load_download_manifest()
    previous_rows = load_previous_processing_rows()

    processing_rows: dict[str, dict[str, str]] = {}

    for source_row in source_rows:
        video_path = source_row["video_path"]
        base_row = blank_processing_row(source_row)
        previous_row = previous_rows.get(video_path)

        if previous_row:
            for field in PROCESSING_MANIFEST_FIELDS:
                if field in previous_row:
                    base_row[field] = previous_row[field]

        # Refresh stable source metadata in case the earlier manifest changed.
        stable_row = blank_processing_row(source_row)
        for field in (
            "video_path",
            "original_video",
            "sequence_path",
            "split",
            "source_label",
            "label",
            "signer_id",
            "recording_id",
            "source",
        ):
            base_row[field] = stable_row[field]

        processing_rows[video_path] = base_row

    requested_labels = None
    if args.labels:
        requested_labels = {
            normalize_model_label(label)
            for label in args.labels
        }

    requested_splits = None
    if args.splits:
        requested_splits = {
            split.lower()
            for split in args.splits
        }

    selected_rows = [
        row
        for row in source_rows
        if (
            requested_labels is None
            or row["model_label"] in requested_labels
        )
        and (
            requested_splits is None
            or row["split"] in requested_splits
        )
    ]

    if args.max_videos is not None:
        selected_rows = selected_rows[: args.max_videos]

    if not selected_rows:
        available_labels = sorted(
            {row["model_label"] for row in source_rows}
        )
        raise ValueError(
            "No videos matched the requested filters. Available labels:\n"
            + ", ".join(available_labels)
        )

    split_counts = Counter(row["split"] for row in selected_rows)
    label_counts = Counter(row["model_label"] for row in selected_rows)

    print("INCLUDE-50 Sequence Preprocessor V4")
    print("--------------------------------")
    print(f"Download manifest: {DOWNLOAD_MANIFEST_PATH}")
    print(f"Selected videos: {len(selected_rows)}")
    print(f"Selected labels: {len(label_counts)}")
    print(f"Sequence shape: ({SEQUENCE_LENGTH}, {FEATURE_COUNT})")
    print(
        "Minimum hand detection: "
        f"{args.min_detection_ratio * 100:.1f}%"
    )
    print(f"Output: {OUTPUT_PATH}")
    print("Official split counts:")
    for split in sorted(split_counts):
        print(f"  {split}: {split_counts[split]}")

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
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
    already_ready_count = 0
    low_detection_count = 0
    source_not_ready_count = 0
    error_count = 0
    interrupted = False

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            for index, source_row in enumerate(selected_rows, start=1):
                video_key = source_row["video_path"]
                processing_row = processing_rows[video_key]
                original_video = resolve_video_path(source_row)
                output_file = output_path_for(source_row)

                prefix = f"[{index}/{len(selected_rows)}]"

                if source_row["status"].strip().upper() != "DOWNLOADED":
                    processing_row.update(
                        {
                            "status": "SOURCE_NOT_READY",
                            "error": (
                                "Download manifest status is "
                                f"{source_row['status']!r}."
                            ),
                        }
                    )
                    print(f"{prefix} SOURCE NOT READY: {video_key}")
                    source_not_ready_count += 1
                    continue

                if not original_video.exists():
                    processing_row.update(
                        {
                            "status": "SOURCE_NOT_READY",
                            "error": "Downloaded video file does not exist.",
                        }
                    )
                    print(f"{prefix} MISSING FILE: {original_video}")
                    source_not_ready_count += 1
                    continue

                expected_size_text = source_row.get(
                    "uncompressed_bytes",
                    "",
                ).strip()

                if expected_size_text:
                    expected_size = int(expected_size_text)
                    actual_size = original_video.stat().st_size

                    if actual_size != expected_size:
                        processing_row.update(
                            {
                                "status": "SOURCE_NOT_READY",
                                "error": (
                                    "Video size mismatch: expected "
                                    f"{expected_size}, found {actual_size}."
                                ),
                            }
                        )
                        print(f"{prefix} SIZE MISMATCH: {video_key}")
                        source_not_ready_count += 1
                        continue

                if (
                    not args.overwrite
                    and valid_existing_sequence(
                        output_file,
                        source_row,
                    )
                ):
                    processing_row.update(
                        metadata_from_existing_sequence(output_file)
                    )
                    print(
                        f"{prefix} READY: {output_file.name} "
                        "(existing)"
                    )
                    already_ready_count += 1
                    continue

                try:
                    sequence, ratios = extract_video_sequence(
                        original_video,
                        landmarker,
                    )

                    if (
                        ratios["detection_ratio"]
                        < args.min_detection_ratio
                    ):
                        processing_row.update(
                            {
                                "detection_ratio": str(
                                    ratios["detection_ratio"]
                                ),
                                "left_hand_ratio": str(
                                    ratios["left_hand_ratio"]
                                ),
                                "right_hand_ratio": str(
                                    ratios["right_hand_ratio"]
                                ),
                                "two_hand_ratio": str(
                                    ratios["two_hand_ratio"]
                                ),
                                "frames": str(SEQUENCE_LENGTH),
                                "features": str(FEATURE_COUNT),
                                "status": "LOW_DETECTION",
                                "error": (
                                    "At least one hand was detected in only "
                                    f"{ratios['detection_ratio'] * 100:.1f}% "
                                    "of sampled frames."
                                ),
                            }
                        )
                        print(
                            f"{prefix} LOW DETECTION: {video_key} — "
                            f"{ratios['detection_ratio'] * 100:.1f}%"
                        )
                        low_detection_count += 1
                        continue

                    save_sequence_atomically(
                        output_file=output_file,
                        sequence=sequence,
                        ratios=ratios,
                        source_row=source_row,
                        original_video=original_video,
                    )
                    update_ready_row(processing_row, ratios)

                    print(
                        f"{prefix} SAVED: {output_file.name} — "
                        f"detection {ratios['detection_ratio'] * 100:.1f}% — "
                        f"both hands {ratios['two_hand_ratio'] * 100:.1f}%"
                    )
                    saved_count += 1

                except Exception as error:
                    processing_row.update(
                        {
                            "status": "ERROR",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    print(f"{prefix} ERROR: {video_key} — {error}")
                    error_count += 1

                if index % 10 == 0:
                    write_processing_manifest(processing_rows)

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user. Saving processing manifest...")

    finally:
        # Refresh every valid output so the manifest remains truthful even
        # after an interrupted or partially resumed run.
        for source_row in source_rows:
            output_file = output_path_for(source_row)

            if valid_existing_sequence(output_file, source_row):
                processing_rows[source_row["video_path"]].update(
                    metadata_from_existing_sequence(output_file)
                )

        write_processing_manifest(processing_rows)

    status_counts = Counter(
        row["status"]
        for row in processing_rows.values()
    )

    print("\nSequence preparation summary")
    print("----------------------------")
    print(f"Saved this run: {saved_count}")
    print(f"Already ready: {already_ready_count}")
    print(f"Low detection this run: {low_detection_count}")
    print(f"Source not ready this run: {source_not_ready_count}")
    print(f"Errors this run: {error_count}")
    print(f"Total ready sequences: {status_counts['READY']}")
    print(f"Pending sequences: {status_counts['PENDING']}")
    print(f"All low detection: {status_counts['LOW_DETECTION']}")
    print(f"All source not ready: {status_counts['SOURCE_NOT_READY']}")
    print(f"All errors: {status_counts['ERROR']}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Manifest: {PROCESSING_MANIFEST_PATH}")

    if interrupted:
        print("\nThe run is resumable. Rerun the same command.")
        return 130

    if (
        low_detection_count
        or source_not_ready_count
        or error_count
    ):
        print(
            "\nSome selected videos were not prepared. "
            "Review their manifest status before training."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
