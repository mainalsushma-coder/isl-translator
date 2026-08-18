ISL CareDesk

Offline Indian Sign Language recognition and sign-to-speech communication assistant

ISL CareDesk is a webcam-based prototype that recognises supported Indian Sign
Language (ISL) gestures, displays the predicted sign and confidence, maps the
result to a clear sentence, and speaks that sentence aloud. The core recognition
and speech pipeline runs locally after the required dependencies and models have
been installed.

Project status

Mode

Status

Vocabulary

Evaluation

Emergency ISL V3

Working MVP

8 signs

97.44% accuracy on the untouched external test set after webcam adaptation

General ISL V4

Under development

50 INCLUDE-50 labels

85.64% accuracy on the official test split; live-webcam adaptation is ongoing

Use app.py for the current stable demonstration. app_general.py is included
for development and testing and should not yet be presented as a production-ready
50-sign live translator.

Emergency signs currently supported

ACCIDENT

CALL

DOCTOR

HELP

HOT

LOSE

PAIN

THIEF

How it works

Webcam / training video
        ↓
OpenCV frame capture
        ↓
MediaPipe two-hand landmark detection
        ↓
48-frame gesture sequence
        ↓
Motion-aware temporal feature extraction
        ↓
Random Forest classification
        ↓
Sign + confidence + top predictions
        ↓
Sentence mapping + offline speech

MediaPipe detects up to two hands and extracts 21 landmarks from each hand. A
complete gesture is represented using 48 evenly sampled frames so the model can
use motion information instead of classifying only one static image.

Technology stack

Python — application runtime and pipeline integration

OpenCV — webcam capture, video reading, frame preparation, drawing and UI

MediaPipe — two-hand detection and 21 landmarks per hand

NumPy — sequence processing, normalisation and temporal feature arrays

Scikit-learn — Random Forest training and evaluation

Joblib — saving and loading trained models

Windows SAPI / pyttsx3 — offline text-to-speech

uv — Python environment and dependency management

Git and GitHub — version control and collaboration

Initial setup using GitHub Desktop

GitHub Desktop clones and manages the repository. It does not directly run the
Python application. After cloning, open the repository terminal and run the
commands below.

1. Install the required tools

GitHub Desktop

uv

A working webcam

Windows 10 or Windows 11 is recommended for the current offline speech setup

Confirm that uv is installed:

uv --version

2. Clone the repository

Open GitHub Desktop.

Select File → Clone repository.

Open the URL tab.

Paste the GitHub repository URL.

Choose a local folder and select Clone.

3. Open the repository terminal

In GitHub Desktop, select:

Repository → Open in Terminal

Depending on the installed version, this may be shown as Open in PowerShell
or Open in Command Prompt.

4. Install/synchronise dependencies

Run this command from the repository folder:

uv sync

5. Verify the required runtime models

Test-Path .\models\hand_landmarker.task
Test-Path .\models\gesture_sequence_adapted_v3.joblib

Both commands should return:

True

6. Run the stable Emergency ISL application

uv run python .\app.py

Application controls

Key

Action

R

Record and recognise one sign

S

Speak the current accepted result

A

Turn automatic speech on or off

C

Clear the current result

H

Clear conversation history

Q

Quit the application

For the best result, use front-facing light and keep the complete hands, wrists
and fingertips visible at chest height. Perform the complete reference gesture
once during recording.

General ISL V4 — development mode

The separate 50-label INCLUDE-50 model has been trained and evaluated, but its
live-webcam generalisation is still being improved. If the required model is
present, contributors can launch it with:

Test-Path .\models\gesture_sequence_include50_v4.joblib
uv run python .\app_general.py

Low-confidence results are intentionally shown as UNCERTAIN. Do not lower the
threshold only to force a prediction; additional multi-signer webcam adaptation
is the correct next step.

Repository data policy

Downloaded datasets, extracted landmark sequences and personal webcam recordings
are intentionally excluded from GitHub. They can be large and may contain
participant data.

The repository should contain source code, dependency files and the small model
files required for running the prototype. It should not contain:

external_data/
data/sequences_v3/
data/sequences_include50_v4/
data/webcam_videos_v3/

Troubleshooting

uv is not recognised

Install uv, close the terminal, reopen it from GitHub Desktop and run:

uv --version

The webcam does not open

Close Camera, Teams, Zoom, Google Meet and other programs using the webcam.

Allow camera access under Windows Settings → Privacy & security → Camera.

Restart the application.

A model file is missing

Pull the latest repository changes in GitHub Desktop and confirm that the model
files listed in the setup section are present. The large source datasets are not
required just to run the stable application.

Speech is unavailable

Run uv sync again and confirm that Windows audio is working. The project uses
Windows SAPI as its primary offline speech backend and pyttsx3 as a fallback.

Current scope and limitations

The current MVP performs isolated-sign recognition, not continuous ISL
sentence translation.

The stable demo supports eight emergency signs.

General 50-label live recognition is under development.

Facial expression and full-body pose are not yet part of the stable classifier.

Performance can be affected by poor lighting, hand occlusion, camera angle,
signer variation and regional ISL variants.

Emergency live-location sharing is a planned feature and is not part of the
current offline MVP. Sending a message would require user consent and network
connectivity.

Datasets and research foundation

Emergency ISL Gesture Video Dataset — Mendeley Data

AI4Bharat INCLUDE dataset and official splits — Hugging Face

INCLUDE original video archive — Zenodo

INCLUDE research paper — ACM Multimedia 2020

Indian Sign Language Research and Training Centre

Development note

This repository is an engineering prototype and research project. Predictions
should not be treated as a replacement for a qualified ISL interpreter in
medical, legal or other high-stakes situations.

