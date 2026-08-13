import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np


HAND_MODEL_PATH = (
    Path(__file__).parent / "models" / "hand_landmarker.task"
)

CLASSIFIER_PATH = (
    Path(__file__).parent / "models" / "gesture_classifier.joblib"
)


# -------------------------------------------------
# Normalize exactly like collect_data.py
# -------------------------------------------------

def normalize_landmarks(landmarks):
    wrist = landmarks[0]

    relative_points = []

    for landmark in landmarks:
        relative_points.append(
            (
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            )
        )

    scale = max(
        max(abs(x), abs(y), abs(z))
        for x, y, z in relative_points
    )

    if scale == 0:
        scale = 1

    features = []

    for x, y, z in relative_points:
        features.extend([
            x / scale,
            y / scale,
            z / scale
        ])

    return features


# -------------------------------------------------
# Load our trained model
# -------------------------------------------------

model_data = joblib.load(CLASSIFIER_PATH)
classifier = model_data["model"]

print("Loaded labels:", model_data["labels"])


# -------------------------------------------------
# MediaPipe configuration
# -------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# Remember recent predictions to reduce flickering
prediction_history = deque(maxlen=7)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")


with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mediapipe_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = time.perf_counter_ns() // 1_000_000

        result = landmarker.detect_for_video(
            mediapipe_image,
            timestamp_ms
        )

        displayed_label = "NO HAND"
        displayed_confidence = 0.0

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            features = normalize_landmarks(landmarks)

            probabilities = classifier.predict_proba(
                np.array([features])
            )[0]

            best_index = int(np.argmax(probabilities))
            confidence = float(probabilities[best_index])
            predicted_label = classifier.classes_[best_index]

            # Ignore uncertain predictions
            if confidence >= 0.75:
                prediction_history.append(predicted_label)

                displayed_label = Counter(
                    prediction_history
                ).most_common(1)[0][0]

                displayed_confidence = confidence
            else:
                displayed_label = "UNCERTAIN"
                displayed_confidence = confidence

            height, width, _ = frame.shape

            # Draw the 21 landmark points
            for landmark in landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        else:
            prediction_history.clear()

        cv2.putText(
            frame,
            f"Gesture: {displayed_label}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {displayed_confidence * 100:.1f}%",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Live Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


camera.release()
cv2.destroyAllWindows()