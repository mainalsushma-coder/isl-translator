import math
import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from prepare_open_dataset import extract_frame_features
from train_sequence_model import sequence_to_model_features


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

HAND_MODEL_PATH = (
    PROJECT_ROOT / "models" / "hand_landmarker.task"
)

CLASSIFIER_PATH = (
    PROJECT_ROOT
    / "models"
    / "gesture_sequence_adapted_v3.joblib"
)

CONFIDENCE_THRESHOLD = 0.55
COUNTDOWN_SECONDS = 2.0


# =========================================================
# Load and validate the model
# =========================================================

if not HAND_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Hand landmark model not found:\n"
        f"{HAND_MODEL_PATH}"
    )

if not CLASSIFIER_PATH.exists():
    raise FileNotFoundError(
        f"Sequence classifier not found:\n"
        f"{CLASSIFIER_PATH}"
    )

model_data = joblib.load(CLASSIFIER_PATH)

classifier = model_data["model"]

sequence_length = int(
    model_data["sequence_length"]
)

frame_feature_count = int(
    model_data["frame_feature_count"]
)

model_feature_count = int(
    model_data["model_feature_count"]
)

minimum_detection_ratio = float(
    model_data["minimum_detection_ratio"]
)

if sequence_length != 48:
    raise ValueError(
        f"Expected sequence length 48, "
        f"model requires {sequence_length}."
    )

if frame_feature_count != 132:
    raise ValueError(
        f"Expected 132 frame features, "
        f"model requires {frame_feature_count}."
    )

if (
    model_data["feature_transform"]
    != "TEMPORAL_SUMMARY_V1"
):
    raise ValueError(
        "Unsupported temporal feature transform."
    )

print("Loaded sequence model")
print("---------------------")
print("Labels:", list(classifier.classes_))
print("Sequence length:", sequence_length)
print("Frame features:", frame_feature_count)
print("Model features:", model_feature_count)
print(
    "Dataset test accuracy:",
    f"{model_data['accuracy'] * 100:.2f}%",
)


# =========================================================
# MediaPipe configuration
# =========================================================

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


# =========================================================
# Camera and recognition state
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

state = "READY"

countdown_end_time = 0.0
captured_features = []
detected_frame_count = 0

displayed_label = "NONE"
displayed_confidence = 0.0
top_predictions = []

last_timestamp_ms = 0


# =========================================================
# Run live recognition
# =========================================================

try:
    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                break

            # Match the external dataset preprocessing.
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
                time.perf_counter_ns() // 1_000_000
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

            frame_features, hand_detected = (
                extract_frame_features(result)
            )

            if len(frame_features) != 132:
                raise ValueError(
                    "Live feature extraction produced "
                    f"{len(frame_features)} values "
                    "instead of 132."
                )

            frame_height, frame_width, _ = (
                frame.shape
            )

            # Draw detected hand landmarks.
            if result.hand_landmarks:
                for hand_landmarks in (
                    result.hand_landmarks
                ):
                    for landmark in hand_landmarks:
                        x = int(
                            landmark.x * frame_width
                        )
                        y = int(
                            landmark.y * frame_height
                        )

                        cv2.circle(
                            frame,
                            (x, y),
                            4,
                            (0, 255, 0),
                            -1,
                        )

            current_time = time.perf_counter()

            # ---------------------------------------------
            # Countdown before recording
            # ---------------------------------------------

            if state == "COUNTDOWN":
                remaining_time = (
                    countdown_end_time - current_time
                )

                if remaining_time <= 0:
                    state = "RECORDING"
                    captured_features = []
                    detected_frame_count = 0

            # ---------------------------------------------
            # Record exactly 48 consecutive frames
            # ---------------------------------------------

            if state == "RECORDING":
                captured_features.append(
                    frame_features.copy()
                )

                if hand_detected:
                    detected_frame_count += 1

                if (
                    len(captured_features)
                    >= sequence_length
                ):
                    sequence = np.asarray(
                        captured_features[
                            :sequence_length
                        ],
                        dtype=np.float32,
                    )

                    detection_ratio = (
                        detected_frame_count
                        / sequence_length
                    )

                    if (
                        detection_ratio
                        < minimum_detection_ratio
                    ):
                        displayed_label = (
                            "LOW HAND DETECTION"
                        )
                        displayed_confidence = 0.0
                        top_predictions = []

                    else:
                        model_features = (
                            sequence_to_model_features(
                                sequence
                            )
                        )

                        if (
                            len(model_features)
                            != model_feature_count
                        ):
                            raise ValueError(
                                "Live temporal features "
                                f"produced "
                                f"{len(model_features)} "
                                "values instead of "
                                f"{model_feature_count}."
                            )

                        probabilities = (
                            classifier.predict_proba(
                                np.asarray(
                                    [model_features],
                                    dtype=np.float32,
                                )
                            )[0]
                        )

                        sorted_indices = np.argsort(
                            probabilities
                        )[::-1]

                        best_index = int(
                            sorted_indices[0]
                        )

                        best_label = str(
                            classifier.classes_[
                                best_index
                            ]
                        )

                        best_confidence = float(
                            probabilities[best_index]
                        )

                        displayed_confidence = (
                            best_confidence
                        )

                        if (
                            best_confidence
                            >= CONFIDENCE_THRESHOLD
                        ):
                            displayed_label = best_label
                        else:
                            displayed_label = (
                                f"UNCERTAIN: {best_label}"
                            )

                        top_predictions = []

                        for index in sorted_indices[:3]:
                            top_predictions.append(
                                (
                                    str(
                                        classifier.classes_[
                                            index
                                        ]
                                    ),
                                    float(
                                        probabilities[index]
                                    ),
                                )
                            )

                    state = "RESULT"

            # =================================================
            # Interface
            # =================================================

            cv2.rectangle(
                frame,
                (0, 0),
                (frame_width, 210),
                (20, 20, 20),
                -1,
            )

            cv2.putText(
                frame,
                f"Status: {state}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            if state == "COUNTDOWN":
                remaining = max(
                    0,
                    math.ceil(
                        countdown_end_time
                        - current_time
                    ),
                )

                cv2.putText(
                    frame,
                    f"Starting in: {remaining}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    "Get into the neutral position",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            elif state == "RECORDING":
                progress = len(captured_features)

                cv2.putText(
                    frame,
                    (
                        f"Recording: {progress}/"
                        f"{sequence_length}"
                    ),
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

                bar_width = int(
                    350
                    * progress
                    / sequence_length
                )

                cv2.rectangle(
                    frame,
                    (20, 100),
                    (370, 125),
                    (255, 255, 255),
                    2,
                )

                cv2.rectangle(
                    frame,
                    (20, 100),
                    (20 + bar_width, 125),
                    (0, 0, 255),
                    -1,
                )

                cv2.putText(
                    frame,
                    "Perform the complete sign now",
                    (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            elif state == "RESULT":
                result_colour = (
                    (0, 255, 0)
                    if displayed_confidence
                    >= CONFIDENCE_THRESHOLD
                    else (0, 255, 255)
                )

                cv2.putText(
                    frame,
                    f"Gesture: {displayed_label}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    result_colour,
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        "Confidence: "
                        f"{displayed_confidence * 100:.1f}%"
                    ),
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                for prediction_number, (
                    label,
                    probability,
                ) in enumerate(top_predictions):
                    cv2.putText(
                        frame,
                        (
                            f"{prediction_number + 1}. "
                            f"{label}: "
                            f"{probability * 100:.1f}%"
                        ),
                        (
                            430,
                            40
                            + prediction_number * 35,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            else:
                cv2.putText(
                    frame,
                    "Press R to recognise a sign",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            cv2.putText(
                frame,
                "R: record | C: clear | Q: quit",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "ISL Sequence Recognition V3",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                state = "COUNTDOWN"
                countdown_end_time = (
                    time.perf_counter()
                    + COUNTDOWN_SECONDS
                )

                captured_features = []
                detected_frame_count = 0
                displayed_label = "NONE"
                displayed_confidence = 0.0
                top_predictions = []

            if key == ord("c"):
                state = "READY"
                captured_features = []
                detected_frame_count = 0
                displayed_label = "NONE"
                displayed_confidence = 0.0
                top_predictions = []

finally:
    camera.release()
    cv2.destroyAllWindows()
