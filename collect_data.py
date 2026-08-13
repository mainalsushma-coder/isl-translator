import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

from feature_extractor import (
    TOTAL_FEATURES,
    extract_two_hand_features
)


# -------------------------------------------------
# Project paths
# -------------------------------------------------

PROJECT_DIR = Path(__file__).parent

MODEL_PATH = (
    PROJECT_DIR /
    "models" /
    "hand_landmarker.task"
)

# New dataset file.
# The original landmarks.csv remains untouched.
DATA_PATH = (
    PROJECT_DIR /
    "data" /
    "landmarks_v2.csv"
)

DATA_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# -------------------------------------------------
# Collection information
# -------------------------------------------------

label = input(
    "Enter gesture label (example TWO_HAND_TEST): "
).strip().upper()

if not label:
    raise ValueError("Label cannot be empty.")


signer_id = input(
    "Enter signer ID (example SUSHMA): "
).strip().upper()

if not signer_id:
    raise ValueError("Signer ID cannot be empty.")


default_session_id = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

session_id = input(
    f"Enter session ID [{default_session_id}]: "
).strip()

if not session_id:
    session_id = default_session_id


target_input = input(
    "Enter number of samples [100]: "
).strip()

if target_input:
    target_samples = int(target_input)
else:
    target_samples = 100

if target_samples <= 0:
    raise ValueError(
        "Number of samples must be greater than zero."
    )


print("\nCollection configuration")
print("------------------------")
print(f"Label: {label}")
print(f"Signer: {signer_id}")
print(f"Session: {session_id}")
print(f"Target samples: {target_samples}")
print(f"Features per sample: {TOTAL_FEATURES}")
print()


# -------------------------------------------------
# MediaPipe setup
# -------------------------------------------------

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,

    # Upgraded from one hand to two hands
    num_hands=2,

    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# -------------------------------------------------
# Prepare CSV schema
# -------------------------------------------------

expected_header = [
    "label",
    "signer_id",
    "session_id"
] + [
    f"feature_{index}"
    for index in range(TOTAL_FEATURES)
]


file_has_data = (
    DATA_PATH.exists() and
    DATA_PATH.stat().st_size > 0
)


if file_has_data:

    with DATA_PATH.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as existing_file:

        reader = csv.reader(existing_file)

        existing_header = next(
            reader,
            None
        )

    if existing_header != expected_header:
        raise RuntimeError(
            "\nThe existing landmarks_v2.csv has a "
            "different feature structure.\n"
            f"Expected {len(expected_header)} columns, "
            f"but found "
            f"{len(existing_header) if existing_header else 0}.\n"
            "Rename or remove the incompatible file "
            "before collecting new data."
        )


# -------------------------------------------------
# Open webcam
# -------------------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open webcam."
    )


# Try to use a reasonable webcam resolution
camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# -------------------------------------------------
# Open CSV file
# -------------------------------------------------

csv_file = DATA_PATH.open(
    "a",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(csv_file)


if not file_has_data:
    writer.writerow(expected_header)
    csv_file.flush()


# -------------------------------------------------
# Collection state
# -------------------------------------------------

sample_count = 0
recording = False
frame_counter = 0
collection_finished = False


def draw_landmarks(
    frame,
    landmarks,
    colour
):
    """
    Draw the 21 landmark points for one hand.
    """

    height, width, _ = frame.shape

    for landmark in landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        cv2.circle(
            frame,
            (x, y),
            5,
            colour,
            -1
        )


# -------------------------------------------------
# Start collection
# -------------------------------------------------

try:

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            success, frame = camera.read()

            if not success:
                print(
                    "Could not read a frame "
                    "from the webcam."
                )
                break

            # Mirror the image like a selfie camera
            frame = cv2.flip(
                frame,
                1
            )

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mediapipe_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = (
                time.perf_counter_ns()
                // 1_000_000
            )

            result = landmarker.detect_for_video(
                mediapipe_image,
                timestamp_ms
            )

            features, hands = (
                extract_two_hand_features(result)
            )

            left_hand = hands["Left"]
            right_hand = hands["Right"]

            # Left hand is shown in green
            if left_hand is not None:
                draw_landmarks(
                    frame,
                    left_hand,
                    (0, 255, 0)
                )

            # Right hand is shown in orange
            if right_hand is not None:
                draw_landmarks(
                    frame,
                    right_hand,
                    (0, 165, 255)
                )

            detected_names = []

            if left_hand is not None:
                detected_names.append("LEFT")

            if right_hand is not None:
                detected_names.append("RIGHT")

            if detected_names:
                hands_text = " + ".join(
                    detected_names
                )
            else:
                hands_text = "NONE"

            # Save one sample every five frames.
            # This reduces the number of nearly
            # identical neighboring frames.
            if (
                recording
                and features is not None
                and frame_counter % 5 == 0
            ):

                row = [
                    label,
                    signer_id,
                    session_id
                ] + features

                writer.writerow(row)

                sample_count += 1

                # Save the data regularly instead of
                # waiting until the program closes.
                if sample_count % 10 == 0:
                    csv_file.flush()

                print(
                    f"{label} | "
                    f"{signer_id} | "
                    f"{sample_count}/"
                    f"{target_samples}"
                )

                if sample_count >= target_samples:

                    csv_file.flush()

                    print(
                        f"\nFinished recording "
                        f"{label}!"
                    )

                    collection_finished = True
                    break

            frame_counter += 1

            if recording:
                status = "RECORDING"
                status_colour = (0, 0, 255)
            else:
                status = "PAUSED"
                status_colour = (0, 255, 255)

            cv2.putText(
                frame,
                f"Label: {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Signer: {signer_id}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                (
                    f"Samples: {sample_count}/"
                    f"{target_samples}"
                ),
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Hands: {hands_text}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Status: {status}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_colour,
                2
            )

            cv2.putText(
                frame,
                "R: record/pause | Q: quit",
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "ISL Dataset Collector V2",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):

                recording = not recording

                if recording:
                    print(
                        "\nRecording started."
                    )
                else:
                    print(
                        "\nRecording paused."
                    )

            elif key == ord("q"):

                print(
                    "\nCollection stopped "
                    "by the user."
                )

                break


finally:

    csv_file.flush()
    csv_file.close()

    camera.release()

    cv2.destroyAllWindows()


print("\nCollection summary")
print("------------------")
print(f"Label: {label}")
print(f"Signer: {signer_id}")
print(f"Session: {session_id}")
print(f"Samples collected: {sample_count}")
print(f"Dataset saved at: {DATA_PATH}")

if collection_finished:
    print("Status: COMPLETE")
else:
    print("Status: STOPPED BEFORE TARGET")