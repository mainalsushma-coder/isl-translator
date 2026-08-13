import time
from pathlib import Path

import cv2
import mediapipe as mp


# -------------------------------------------------
# 1. Location of MediaPipe's trained hand model
# -------------------------------------------------

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"


# Connections between MediaPipe's 21 hand points
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # Little finger
    (0, 17)
]


# -------------------------------------------------
# 2. Configure MediaPipe
# -------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# -------------------------------------------------
# 3. Open webcam
# -------------------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open the webcam.")


# -------------------------------------------------
# 4. Detect hand landmarks
# -------------------------------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        # Mirror the webcam
        frame = cv2.flip(frame, 1)

        # OpenCV gives BGR; MediaPipe expects RGB
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

        height, width, _ = frame.shape

        # Process every detected hand
        for hand_index, landmarks in enumerate(result.hand_landmarks):

            points = []

            # Convert MediaPipe coordinates into screen pixels
            for landmark in landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append((x, y))

            # Draw the green connections
            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )

            # Draw the red landmark points
            for point in points:
                cv2.circle(
                    frame,
                    point,
                    5,
                    (0, 0, 255),
                    -1
                )

            # Display Left or Right hand
            hand_name = result.handedness[hand_index][0].category_name

            wrist_x, wrist_y = points[0]

            cv2.putText(
                frame,
                hand_name,
                (wrist_x, wrist_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow("ISL Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


camera.release()
cv2.destroyAllWindows()