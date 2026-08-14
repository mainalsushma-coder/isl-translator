import math
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyttsx3

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
PANEL_WIDTH = 480
HISTORY_LIMIT = 5


# =========================================================
# Friendly English output
# =========================================================

GESTURE_MESSAGES = {
    "ACCIDENT": "There has been an accident.",
    "CALL": "Please call for assistance.",
    "DOCTOR": "I need a doctor.",
    "HELP": "I need help.",
    "HOT": "It feels hot.",
    "LOSE": "I have lost something.",
    "PAIN": "I am in pain.",
    "THIEF": "There is a thief.",
}


# =========================================================
# Offline speech worker
# =========================================================

speech_queue = queue.Queue()

speech_state = {
    "ready": False,
    "error": "",
}


def speech_worker():
    try:
        # Required when SAPI is started in a background
        # thread on Windows.
        try:
            import pythoncom

            pythoncom.CoInitialize()
        except ImportError:
            pythoncom = None

        engine = pyttsx3.init()

        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)

        speech_state["ready"] = True

        while True:
            message = speech_queue.get()

            if message is None:
                break

            engine.say(message)
            engine.runAndWait()

            speech_queue.task_done()

    except Exception as error:
        speech_state["error"] = str(error)

    finally:
        try:
            if pythoncom is not None:
                pythoncom.CoUninitialize()
        except Exception:
            pass


speech_thread = threading.Thread(
    target=speech_worker,
    daemon=True,
)

speech_thread.start()


# =========================================================
# Interface helpers
# =========================================================

def draw_wrapped_text(
    image,
    text,
    x,
    y,
    maximum_width,
    font_scale=0.60,
    colour=(255, 255, 255),
    thickness=1,
    line_height=27,
):
    words = text.split()
    current_line = ""
    lines = []

    for word in words:
        test_line = (
            word
            if not current_line
            else f"{current_line} {word}"
        )

        text_width = cv2.getTextSize(
            test_line,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )[0][0]

        if text_width <= maximum_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)

            current_line = word

    if current_line:
        lines.append(current_line)

    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            colour,
            thickness,
            cv2.LINE_AA,
        )

        y += line_height

    return y


# =========================================================
# Load model
# =========================================================

if not HAND_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Hand model not found:\n{HAND_MODEL_PATH}"
    )

if not CLASSIFIER_PATH.exists():
    raise FileNotFoundError(
        f"Adapted classifier not found:\n"
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
        f"Expected 48 frames, model requires "
        f"{sequence_length}."
    )

if frame_feature_count != 132:
    raise ValueError(
        f"Expected 132 features per frame, "
        f"model requires {frame_feature_count}."
    )

if (
    model_data["feature_transform"]
    != "TEMPORAL_SUMMARY_V1"
):
    raise ValueError(
        "Unsupported feature transformation."
    )

print("ISL CareDesk")
print("------------")
print("Model:", CLASSIFIER_PATH.name)
print("Labels:", list(classifier.classes_))
print(
    "External test accuracy:",
    f"{model_data['accuracy'] * 100:.2f}%",
)


# =========================================================
# MediaPipe setup
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
# Camera and application state
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

state = "READY"
countdown_end_time = 0.0

captured_features = []
detected_frame_count = 0

displayed_label = "NONE"
displayed_message = (
    "Press R and perform an emergency sign."
)
displayed_confidence = 0.0

top_predictions = []
conversation_history = deque(
    maxlen=HISTORY_LIMIT
)

automatic_speech = True
last_timestamp_ms = 0


# =========================================================
# Main application
# =========================================================

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

            frame_features, hand_detected = (
                extract_frame_features(result)
            )

            if (
                len(frame_features)
                != frame_feature_count
            ):
                raise ValueError(
                    "Live feature count does not match "
                    "the trained model."
                )

            frame_height, frame_width, _ = (
                frame.shape
            )

            # Draw hand landmarks.
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

            # -------------------------------------------------
            # Countdown
            # -------------------------------------------------

            if state == "COUNTDOWN":
                if current_time >= countdown_end_time:
                    state = "RECORDING"
                    captured_features = []
                    detected_frame_count = 0

            # -------------------------------------------------
            # Record and recognise
            # -------------------------------------------------

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
                        displayed_label = "NO HAND"
                        displayed_message = (
                            "Keep your complete hands "
                            "visible and try again."
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
                                "Temporal feature count "
                                "does not match the model."
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

                        if (
                            best_confidence
                            >= CONFIDENCE_THRESHOLD
                        ):
                            displayed_label = best_label

                            displayed_message = (
                                GESTURE_MESSAGES.get(
                                    best_label,
                                    best_label.title(),
                                )
                            )

                            history_time = time.strftime(
                                "%H:%M:%S"
                            )

                            conversation_history.appendleft(
                                (
                                    history_time,
                                    best_label,
                                    displayed_message,
                                    best_confidence,
                                )
                            )

                            if automatic_speech:
                                speech_queue.put(
                                    displayed_message
                                )

                        else:
                            displayed_label = "UNCERTAIN"

                            displayed_message = (
                                f"Possible sign: "
                                f"{best_label}. "
                                f"Please try again."
                            )

                    state = "RESULT"

            # =================================================
            # Build interface canvas
            # =================================================

            canvas_height = max(
                frame_height,
                700,
            )

            canvas_width = (
                frame_width + PANEL_WIDTH
            )

            canvas = np.zeros(
                (
                    canvas_height,
                    canvas_width,
                    3,
                ),
                dtype=np.uint8,
            )

            canvas[:] = (14, 17, 22)

            canvas[
                :frame_height,
                :frame_width,
            ] = frame

            panel_x = frame_width

            cv2.rectangle(
                canvas,
                (panel_x, 0),
                (canvas_width, canvas_height),
                (25, 31, 40),
                -1,
            )

            # -------------------------------------------------
            # Camera status
            # -------------------------------------------------

            cv2.putText(
                canvas,
                "ISL CareDesk",
                (panel_x + 25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (56, 212, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                canvas,
                f"Status: {state}",
                (panel_x + 25, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (210, 220, 230),
                1,
                cv2.LINE_AA,
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
                    canvas,
                    f"Starting in {remaining}",
                    (panel_x + 25, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.82,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            elif state == "RECORDING":
                recording_progress = len(
                    captured_features
                )

                cv2.putText(
                    canvas,
                    (
                        f"Recording "
                        f"{recording_progress}/"
                        f"{sequence_length}"
                    ),
                    (panel_x + 25, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 90, 255),
                    2,
                    cv2.LINE_AA,
                )

                progress_width = int(
                    400
                    * recording_progress
                    / sequence_length
                )

                cv2.rectangle(
                    canvas,
                    (panel_x + 25, 140),
                    (panel_x + 425, 160),
                    (100, 110, 120),
                    2,
                )

                cv2.rectangle(
                    canvas,
                    (panel_x + 25, 140),
                    (
                        panel_x
                        + 25
                        + progress_width,
                        160,
                    ),
                    (0, 90, 255),
                    -1,
                )

            # -------------------------------------------------
            # Recognition result
            # -------------------------------------------------

            result_colour = (
                (30, 220, 80)
                if (
                    displayed_label
                    not in {
                        "NONE",
                        "UNCERTAIN",
                        "NO HAND",
                    }
                )
                else (0, 220, 255)
            )

            cv2.putText(
                canvas,
                f"Sign: {displayed_label}",
                (panel_x + 25, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                result_colour,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                canvas,
                (
                    "Confidence: "
                    f"{displayed_confidence * 100:.1f}%"
                ),
                (panel_x + 25, 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (210, 220, 230),
                1,
                cv2.LINE_AA,
            )

            message_end_y = draw_wrapped_text(
                canvas,
                displayed_message,
                panel_x + 25,
                285,
                PANEL_WIDTH - 50,
                font_scale=0.65,
                colour=(255, 255, 255),
                thickness=1,
                line_height=30,
            )

            # -------------------------------------------------
            # Top predictions
            # -------------------------------------------------

            prediction_y = max(
                message_end_y + 15,
                350,
            )

            cv2.putText(
                canvas,
                "Top predictions",
                (panel_x + 25, prediction_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (56, 212, 255),
                2,
                cv2.LINE_AA,
            )

            prediction_y += 32

            for prediction_number, (
                label,
                probability,
            ) in enumerate(top_predictions):
                cv2.putText(
                    canvas,
                    (
                        f"{prediction_number + 1}. "
                        f"{label}: "
                        f"{probability * 100:.1f}%"
                    ),
                    (
                        panel_x + 25,
                        prediction_y,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (210, 220, 230),
                    1,
                    cv2.LINE_AA,
                )

                prediction_y += 28

            # -------------------------------------------------
            # Controls
            # -------------------------------------------------

            controls_y = 485

            cv2.putText(
                canvas,
                "Controls",
                (panel_x + 25, controls_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (56, 212, 255),
                2,
                cv2.LINE_AA,
            )

            controls = [
                "R  Recognise a sign",
                "S  Speak result",
                "A  Toggle automatic speech",
                "C  Clear current result",
                "H  Clear history",
                "Q  Quit",
            ]

            controls_y += 30

            for control in controls:
                cv2.putText(
                    canvas,
                    control,
                    (panel_x + 25, controls_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (200, 210, 220),
                    1,
                    cv2.LINE_AA,
                )

                controls_y += 25

            speech_text = (
                "Auto speech: ON"
                if automatic_speech
                else "Auto speech: OFF"
            )

            if speech_state["error"]:
                speech_text = "Speech unavailable"

            cv2.putText(
                canvas,
                speech_text,
                (panel_x + 250, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (
                    (30, 220, 80)
                    if not speech_state["error"]
                    else (0, 80, 255)
                ),
                1,
                cv2.LINE_AA,
            )

            # -------------------------------------------------
            # Conversation history
            # -------------------------------------------------

            history_y = frame_height + 35

            if history_y < canvas_height - 20:
                cv2.putText(
                    canvas,
                    "Recent conversation",
                    (20, history_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (56, 212, 255),
                    2,
                    cv2.LINE_AA,
                )

                history_y += 30

                for (
                    history_time,
                    history_label,
                    history_message,
                    history_confidence,
                ) in conversation_history:
                    history_text = (
                        f"{history_time}  "
                        f"{history_label}  "
                        f"{history_message}  "
                        f"({history_confidence * 100:.1f}%)"
                    )

                    cv2.putText(
                        canvas,
                        history_text,
                        (20, history_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (210, 220, 230),
                        1,
                        cv2.LINE_AA,
                    )

                    history_y += 25

                    if (
                        history_y
                        >= canvas_height - 15
                    ):
                        break

            cv2.imshow(
                "ISL CareDesk Prototype",
                canvas,
            )

            # =================================================
            # Keyboard controls
            # =================================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if (
                key == ord("r")
                and state != "RECORDING"
            ):
                state = "COUNTDOWN"

                countdown_end_time = (
                    time.perf_counter()
                    + COUNTDOWN_SECONDS
                )

                captured_features = []
                detected_frame_count = 0

                displayed_label = "NONE"
                displayed_message = (
                    "Get ready in the neutral position."
                )
                displayed_confidence = 0.0
                top_predictions = []

            if key == ord("s"):
                if (
                    displayed_label
                    not in {
                        "NONE",
                        "UNCERTAIN",
                        "NO HAND",
                    }
                ):
                    speech_queue.put(
                        displayed_message
                    )

            if key == ord("a"):
                automatic_speech = (
                    not automatic_speech
                )

            if key == ord("c"):
                state = "READY"
                captured_features = []
                detected_frame_count = 0
                displayed_label = "NONE"
                displayed_message = (
                    "Press R and perform an "
                    "emergency sign."
                )
                displayed_confidence = 0.0
                top_predictions = []

            if key == ord("h"):
                conversation_history.clear()

finally:
    camera.release()
    cv2.destroyAllWindows()

    speech_queue.put(None)
    speech_thread.join(timeout=2.0)