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
from train_include50_model import sequence_to_model_features


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
    / "gesture_sequence_include50_v4.joblib"
)

CONFIDENCE_THRESHOLD = 0.35
CONFIDENCE_MARGIN_THRESHOLD = 0.08
COUNTDOWN_SECONDS = 2.0
# INCLUDE-50 HELLO source clips have a median duration of 2.60 seconds.
# Capture the complete live gesture for the same duration, then sample the
# recording evenly into the model's required 48 frames.
RECORDING_SECONDS = 2.60
PANEL_WIDTH = 480
HISTORY_LIMIT = 5

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


# =========================================================
# Friendly English output
# =========================================================

GESTURE_MESSAGES = {
    "BANK": "Bank.",
    "BIG_LARGE": "Big or large.",
    "BIRD": "Bird.",
    "BLACK": "Black.",
    "BOY": "Boy.",
    "BROTHER": "Brother.",
    "CAR": "Car.",
    "CELL_PHONE": "Cell phone.",
    "COURT": "Court.",
    "COW": "Cow.",
    "DEATH": "Death.",
    "DOG": "Dog.",
    "DRY": "Dry.",
    "ELECTION": "Election.",
    "FALL": "Fall.",
    "FAN": "Fan.",
    "FATHER": "Father.",
    "GIRL": "Girl.",
    "GOOD": "Good.",
    "GOOD_MORNING": "Good morning.",
    "HAPPY": "Happy.",
    "HAT": "Hat.",
    "HELLO": "Hello.",
    "HOT": "Hot.",
    "HOUSE": "House.",
    "I": "I.",
    "IT": "It.",
    "LONG": "Long.",
    "LOUD": "Loud.",
    "MONDAY": "Monday.",
    "NEW": "New.",
    "PAINT": "Paint.",
    "PEN": "Pen.",
    "PRIEST": "Priest.",
    "QUIET": "Quiet.",
    "RED": "Red.",
    "SHOES": "Shoes.",
    "SHORT": "Short.",
    "SMALL_LITTLE": "Small or little.",
    "STORE_OR_SHOP": "Store or shop.",
    "SUMMER": "Summer.",
    "TEACHER": "Teacher.",
    "THANK_YOU": "Thank you.",
    "TIME": "Time.",
    "TRAIN_TICKET": "Train ticket.",
    "T_SHIRT": "T-shirt.",
    "WHITE": "White.",
    "WINDOW": "Window.",
    "YEAR": "Year.",
    "YOU_PLURAL": "You all.",
}


# =========================================================
# Offline speech worker
# =========================================================

speech_queue = queue.Queue()

speech_state = {
    "ready": False,
    "speaking": False,
    "backend": "",
    "last_message": "",
    "error": "",
}


def speech_worker():
    pythoncom = None
    sapi_voice = None

    try:
        # Every background thread that uses Windows speech
        # must initialise COM inside that same thread.
        try:
            import pythoncom

            pythoncom.CoInitialize()
        except ImportError:
            pythoncom = None

        # pyttsx3 can occasionally stop producing audio after
        # its first runAndWait() call on Windows. Use Windows'
        # native offline SAPI voice as the primary backend.
        try:
            from win32com.client import Dispatch

            sapi_voice = Dispatch("SAPI.SpVoice")
            sapi_voice.Rate = 0
            sapi_voice.Volume = 100
            speech_state["backend"] = "WINDOWS_SAPI"
        except Exception:
            # pyttsx3 remains available as a fallback. A fresh
            # engine is created for every message so a stalled
            # event loop cannot block later messages.
            sapi_voice = None
            speech_state["backend"] = "PYTTSX3_FALLBACK"

        speech_state["ready"] = True

        while True:
            message = speech_queue.get()

            try:
                if message is None:
                    break

                message = str(message).strip()

                if not message:
                    continue

                speech_state["speaking"] = True
                speech_state["error"] = ""

                if sapi_voice is not None:
                    # Speak synchronously inside this worker.
                    # The camera/UI thread remains responsive.
                    sapi_voice.Speak(message)
                else:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", 165)
                    engine.setProperty("volume", 1.0)
                    engine.say(message)
                    engine.runAndWait()
                    engine.stop()

                speech_state["last_message"] = message

            except Exception as error:
                # Keep the worker alive so one speech failure
                # does not disable every later result.
                speech_state["error"] = str(error)

            finally:
                speech_state["speaking"] = False
                speech_queue.task_done()

    except Exception as error:
        speech_state["error"] = str(error)
        speech_state["ready"] = False

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
        f"INCLUDE-50 classifier not found:\n"
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

print("ISL CareDesk - General ISL V4")
print("-----------------------------")
print("Model:", CLASSIFIER_PATH.name)
print("Labels:", list(classifier.classes_))
print(
    "Official-split test accuracy:",
    f"{model_data['accuracy'] * 100:.2f}%",
)
print(
    "IMPORTANT: Live webcam performance must be "
    "tested separately."
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
recording_start_time = 0.0
recording_end_time = 0.0

captured_features = []
captured_detection_values = []

displayed_label = "NONE"
displayed_message = (
    "Press R and perform a General ISL sign."
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

            # Draw MediaPipe-style hand connections and landmarks.
            if result.hand_landmarks:
                for hand_landmarks in (
                    result.hand_landmarks
                ):
                    pixel_points = []

                    for landmark in hand_landmarks:
                        x = int(
                            landmark.x * frame_width
                        )
                        y = int(
                            landmark.y * frame_height
                        )

                        pixel_points.append((x, y))

                    for start_index, end_index in (
                        HAND_CONNECTIONS
                    ):
                        cv2.line(
                            frame,
                            pixel_points[start_index],
                            pixel_points[end_index],
                            (0, 220, 0),
                            2,
                            cv2.LINE_AA,
                        )

                    for x, y in pixel_points:
                        cv2.circle(
                            frame,
                            (x, y),
                            4,
                            (0, 0, 255),
                            -1,
                            cv2.LINE_AA,
                        )

            current_time = time.perf_counter()

            # -------------------------------------------------
            # Countdown
            # -------------------------------------------------

            if state == "COUNTDOWN":
                if current_time >= countdown_end_time:
                    state = "RECORDING"
                    captured_features = []
                    captured_detection_values = []
                    recording_start_time = current_time
                    recording_end_time = (
                        current_time + RECORDING_SECONDS
                    )
                    displayed_message = (
                        "Perform the complete sign once, "
                        "then hold the ending position."
                    )

            # -------------------------------------------------
            # Record and recognise
            # -------------------------------------------------

            if state == "RECORDING":
                captured_features.append(
                    frame_features.copy()
                )

                captured_detection_values.append(
                    float(hand_detected)
                )

                if (
                    current_time >= recording_end_time
                ):
                    if not captured_features:
                        raise RuntimeError(
                            "No webcam frames were captured."
                        )

                    # Match INCLUDE-50 preprocessing: sample 48 positions
                    # evenly over the complete recording, regardless of the
                    # webcam's actual frame rate.
                    selected_indices = np.rint(
                        np.linspace(
                            0,
                            len(captured_features) - 1,
                            sequence_length,
                        )
                    ).astype(int)

                    sequence = np.asarray(
                        [
                            captured_features[index]
                            for index in selected_indices
                        ],
                        dtype=np.float32,
                    )

                    detection_ratio = (
                        float(
                            np.mean(
                                [
                                    captured_detection_values[index]
                                    for index in selected_indices
                                ]
                            )
                        )
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

                        second_confidence = (
                            float(
                                probabilities[
                                    int(sorted_indices[1])
                                ]
                            )
                            if len(sorted_indices) > 1
                            else 0.0
                        )

                        confidence_margin = (
                            best_confidence
                            - second_confidence
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
                            and confidence_margin
                            >= CONFIDENCE_MARGIN_THRESHOLD
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
                                f"Please perform it again."
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
                f"GENERAL ISL | {state}",
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
                elapsed_recording_time = max(
                    0.0,
                    current_time - recording_start_time,
                )

                recording_progress = min(
                    1.0,
                    elapsed_recording_time
                    / RECORDING_SECONDS,
                )

                remaining_recording_time = max(
                    0.0,
                    recording_end_time - current_time,
                )

                cv2.putText(
                    canvas,
                    (
                        f"Recording: "
                        f"{remaining_recording_time:.1f}s left"
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
                "ISL CareDesk - General ISL V4",
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
                captured_detection_values = []
                recording_start_time = 0.0
                recording_end_time = 0.0

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
                captured_detection_values = []
                recording_start_time = 0.0
                recording_end_time = 0.0
                displayed_label = "NONE"
                displayed_message = (
                    "Press R and perform an "
                    "INCLUDE-50 sign."
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