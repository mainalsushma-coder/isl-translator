"""Run live two-hand gesture recognition using the V2 classifier."""

import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from feature_extractor import TOTAL_FEATURES, extract_two_hand_features


PROJECT_DIR = Path(__file__).parent

HAND_MODEL_PATH = (
    PROJECT_DIR / "models" / "hand_landmarker.task"
)

CLASSIFIER_PATH = (
    PROJECT_DIR / "models" / "gesture_classifier_v2.joblib"
)

CONFIDENCE_THRESHOLD = 0.75
HISTORY_SIZE = 7
MIN_STABLE_VOTES = 4


def load_classifier():
    """Load the V2 model and verify that its feature schema is compatible."""

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"V2 classifier not found: {CLASSIFIER_PATH}\n"
            "Run: uv run python train_model.py"
        )

    model_data = joblib.load(CLASSIFIER_PATH)

    if not isinstance(model_data, dict) or "model" not in model_data:
        raise ValueError(
            "The classifier file does not contain the expected model package."
        )

    saved_feature_count = model_data.get("feature_count")

    if saved_feature_count != TOTAL_FEATURES:
        raise ValueError(
            "Model and feature extractor are incompatible.\n"
            f"Model expects: {saved_feature_count}\n"
            f"Extractor creates: {TOTAL_FEATURES}\n"
            "Retrain the V2 model using the current train_model.py."
        )

    classifier = model_data["model"]
    classifier_feature_count = getattr(
        classifier,
        "n_features_in_",
        None,
    )

    if classifier_feature_count != TOTAL_FEATURES:
        raise ValueError(
            "The trained classifier itself has the wrong feature count.\n"
            f"Classifier expects: {classifier_feature_count}\n"
            f"Live pipeline creates: {TOTAL_FEATURES}"
        )

    labels = model_data.get(
        "labels",
        classifier.classes_.tolist(),
    )

    print("ISL Live Recognition V2")
    print("-----------------------")
    print(f"Classifier: {CLASSIFIER_PATH}")
    print(f"Loaded labels: {labels}")
    print(f"Expected features: {saved_feature_count}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print(
        "Evaluation split used during training: "
        f"{model_data.get('split_strategy', 'UNKNOWN')}"
    )

    if not model_data.get("leakage_safe_evaluation", False):
        print(
            "WARNING: This is a development model. Its reported test "
            "accuracy was not signer-disjoint."
        )

    print("\nPress Q in the camera window to quit.\n")

    return classifier


def draw_hand(frame, landmarks, colour, hand_name):
    """Draw one detected hand and label it as LEFT or RIGHT."""

    height, width, _ = frame.shape

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            5,
            colour,
            -1,
        )

    wrist = landmarks[0]
    wrist_x = int(wrist.x * width)
    wrist_y = int(wrist.y * height)

    cv2.putText(
        frame,
        hand_name.upper(),
        (wrist_x + 10, max(25, wrist_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        colour,
        2,
    )


def stable_prediction(prediction_history, current_confidence):
    """Return a prediction only when enough recent frames agree."""

    if not prediction_history:
        return "UNCERTAIN", current_confidence

    vote_counts = Counter(
        label
        for label, _ in prediction_history
    )

    winning_label, winning_votes = vote_counts.most_common(1)[0]

    if (
        winning_label == "UNCERTAIN"
        or winning_votes < MIN_STABLE_VOTES
    ):
        return "UNCERTAIN", current_confidence

    winning_confidences = [
        confidence
        for label, confidence in prediction_history
        if label == winning_label
    ]

    average_confidence = float(
        np.mean(winning_confidences)
    )

    return winning_label, average_confidence


def main():
    classifier = load_classifier()

    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe hand model not found: {HAND_MODEL_PATH}"
        )

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
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    prediction_history = deque(maxlen=HISTORY_SIZE)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_timestamp_ms = 0

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                # Mirror the frame in exactly the same way as collect_data.py.
                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mediapipe_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = time.perf_counter_ns() // 1_000_000

                # VIDEO mode requires monotonically increasing timestamps.
                timestamp_ms = max(
                    timestamp_ms,
                    last_timestamp_ms + 1,
                )
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    mediapipe_image,
                    timestamp_ms,
                )

                features, hands = extract_two_hand_features(result)

                left_hand = hands["Left"]
                right_hand = hands["Right"]

                if left_hand is not None:
                    draw_hand(
                        frame,
                        left_hand,
                        (0, 255, 0),
                        "LEFT",
                    )

                if right_hand is not None:
                    draw_hand(
                        frame,
                        right_hand,
                        (0, 165, 255),
                        "RIGHT",
                    )

                detected_hand_names = []

                if left_hand is not None:
                    detected_hand_names.append("LEFT")

                if right_hand is not None:
                    detected_hand_names.append("RIGHT")

                if detected_hand_names:
                    hands_text = " + ".join(detected_hand_names)
                else:
                    hands_text = "NONE"

                displayed_label = "NO HAND"
                displayed_confidence = 0.0
                raw_text = "NONE"

                if features is not None:
                    if len(features) != TOTAL_FEATURES:
                        raise RuntimeError(
                            f"Expected {TOTAL_FEATURES} live features, "
                            f"received {len(features)}."
                        )

                    feature_array = np.asarray(
                        [features],
                        dtype=np.float32,
                    )

                    probabilities = classifier.predict_proba(
                        feature_array
                    )[0]

                    best_index = int(np.argmax(probabilities))
                    confidence = float(probabilities[best_index])
                    predicted_label = str(
                        classifier.classes_[best_index]
                    )

                    raw_text = (
                        f"{predicted_label} "
                        f"({confidence * 100:.1f}%)"
                    )

                    if confidence >= CONFIDENCE_THRESHOLD:
                        candidate_label = predicted_label
                    else:
                        candidate_label = "UNCERTAIN"

                    prediction_history.append(
                        (candidate_label, confidence)
                    )

                    (
                        displayed_label,
                        displayed_confidence,
                    ) = stable_prediction(
                        prediction_history,
                        confidence,
                    )
                else:
                    prediction_history.clear()

                if displayed_label == "NO HAND":
                    label_colour = (180, 180, 180)
                elif displayed_label == "UNCERTAIN":
                    label_colour = (0, 255, 255)
                else:
                    label_colour = (0, 255, 0)

                cv2.putText(
                    frame,
                    f"Gesture: {displayed_label}",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    label_colour,
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        "Stable confidence: "
                        f"{displayed_confidence * 100:.1f}%"
                    ),
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Hands: {hands_text}",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Raw prediction: {raw_text}",
                    (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Threshold: {CONFIDENCE_THRESHOLD * 100:.0f}% "
                        f"| Votes: {MIN_STABLE_VOTES}/{HISTORY_SIZE}"
                    ),
                    (20, 205),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    "Press Q to quit",
                    (20, 245),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(
                    "ISL Live Gesture Recognition V2",
                    frame,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()