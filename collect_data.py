import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
DATA_PATH = Path(__file__).parent / "data" / "landmarks.csv"

DATA_PATH.parent.mkdir(exist_ok=True)

label = input("Enter gesture label (example OPEN_PALM): ").strip().upper()

if not label:
    raise ValueError("Label cannot be empty.")


# -------------------------------------------------
# Normalize landmarks
# -------------------------------------------------

def normalize_landmarks(landmarks):
    """
    Convert 21 points into 63 normalized values:
    21 landmarks × (x, y, z)
    """

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

    # Normalize for different distances from the camera
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
# MediaPipe setup
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
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")


# -------------------------------------------------
# CSV setup
# -------------------------------------------------

file_already_exists = DATA_PATH.exists()

csv_file = DATA_PATH.open("a", newline="")
writer = csv.writer(csv_file)

if not file_already_exists:
    header = ["label"] + [f"feature_{i}" for i in range(63)]
    writer.writerow(header)


# -------------------------------------------------
# Collect samples
# -------------------------------------------------

sample_count = 0
target_samples = 200
recording = False
frame_counter = 0

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

        height, width, _ = frame.shape

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            # Draw landmark points
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

            # Save one sample every third frame
            if recording and frame_counter % 3 == 0:
                features = normalize_landmarks(landmarks)
                writer.writerow([label] + features)
                sample_count += 1

                print(f"{label}: {sample_count}/{target_samples}")

                if sample_count >= target_samples:
                    print(f"\nFinished recording {label}!")
                    break

        frame_counter += 1

        status = "RECORDING" if recording else "PAUSED"

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
            f"Samples: {sample_count}/{target_samples}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if recording else (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "R: record/pause | Q: quit",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("ISL Dataset Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            recording = not recording

        elif key == ord("q"):
            break


csv_file.close()
camera.release()
cv2.destroyAllWindows()

print(f"Dataset saved at: {DATA_PATH}")