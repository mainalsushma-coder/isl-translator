# ISL CareDesk

An offline, webcam-based Indian Sign Language emergency communication prototype.

ISL CareDesk recognises supported emergency signs, displays the predicted sign and confidence, converts it into a clear English sentence, and speaks the result aloud using offline text-to-speech.

> The current stable demonstration is `app.py`, supporting 8 emergency signs.

## Current Status

| Component | Status |
|---|---|
| Emergency sign recognition | Working prototype |
| Supported signs | 8 |
| Webcam recognition | Working |
| Two-hand tracking | Working |
| Sentence generation | Working |
| Offline speech | Working |
| Internet required while running | No |
| External test accuracy | 97.44% |

The reported accuracy is from an untouched external test set after webcam adaptation. It should not be interpreted as universal real-world ISL accuracy.

## Supported Emergency Signs

| Sign | Application output |
|---|---|
| `ACCIDENT` | There has been an accident. |
| `CALL` | Please call for assistance. |
| `DOCTOR` | I need a doctor. |
| `HELP` | I need help. |
| `HOT` | It feels hot. |
| `LOSE` | I have lost something. |
| `PAIN` | I am in pain. |
| `THIEF` | There is a thief. |

## How It Works

```text
Webcam video
    ↓
OpenCV frame capture
    ↓
MediaPipe two-hand tracking
    ↓
21 landmarks from each detected hand
    ↓
48-frame gesture sequence
    ↓
Motion-aware temporal feature extraction
    ↓
Random Forest classification
    ↓
Predicted sign + confidence + top predictions
    ↓
Natural English sentence
    ↓
Offline speech using Windows SAPI / pyttsx3
```

The application records 48 frames for each attempt. This allows the model to analyse hand movement across time instead of classifying only one static image.

## Technology Stack

- **Python** — application runtime and pipeline integration
- **OpenCV** — webcam access, frame processing, interface rendering and keyboard controls
- **MediaPipe** — detection and tracking of up to two hands
- **NumPy** — landmark arrays and sequence processing
- **scikit-learn** — Random Forest classifier
- **Joblib** — saving and loading the trained model
- **pyttsx3 / Windows SAPI** — offline text-to-speech
- **uv** — Python environment and dependency management

## System Requirements

The prototype has been tested on Windows.

You will need:

- Windows 10 or Windows 11
- A working webcam
- Speakers or headphones
- Git
- `uv`
- Adequate lighting with the hands clearly visible

> Run this application locally. GitHub stores the code, but the GitHub website itself cannot open your computer's webcam and application window.

## Initial Setup

### 1. Open PowerShell

Open PowerShell or the VS Code terminal.

### 2. Install Git

Download Git from:

https://git-scm.com/download/win

After installation, confirm that it works:

```powershell
git --version
```

### 3. Install uv

If `uv` is not already installed, run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell, then confirm:

```powershell
uv --version
```

### 4. Clone the Repository

```powershell
cd D:\projects
git clone https://github.com/mainalsushma-coder/isl-translator.git
cd .\isl-translator
```

You can use another folder instead of `D:\projects` if required.

### 5. Install Project Dependencies

Run this command from inside the repository:

```powershell
uv sync
```

`uv` will create the project environment and install the required Python packages.

### 6. Confirm the Required Models Exist

Run:

```powershell
Test-Path .\models\hand_landmarker.task
Test-Path .\models\gesture_sequence_adapted_v3.joblib
```

Both commands must return:

```text
True
```

### 7. Start ISL CareDesk

```powershell
uv run python .\app.py
```

The webcam window named **ISL CareDesk Prototype** should open.

## Running an Existing Clone

If the repository is already on your computer, use:

```powershell
cd D:\projects\isl-ai-translator
git switch main
git pull origin main
uv sync
uv run python .\app.py
```

Change the folder path if the project is stored somewhere else.

## Using the Application

1. Allow camera access if Windows requests permission.
2. Sit where light falls toward your face and hands.
3. Keep your hands, wrists and fingertips inside the camera frame.
4. Click the application window so that it receives keyboard input.
5. Press `R`.
6. Wait for the two-second countdown.
7. Perform the complete emergency sign during the 48-frame recording.
8. Wait for the predicted sign, confidence and sentence.
9. If automatic speech is enabled, the sentence will be spoken aloud.

## Keyboard Controls

| Key | Action |
|---|---|
| `R` | Record and recognise a sign |
| `S` | Speak the current result again |
| `A` | Enable or disable automatic speech |
| `C` | Clear the current result |
| `H` | Clear conversation history |
| `Q` | Close the application |

## Recognition Tips

For better results:

- Use bright, even lighting.
- Keep the complete hands visible.
- Avoid covering one hand with the other unnecessarily.
- Keep hands approximately around chest height.
- Begin from a neutral position.
- Perform the complete gesture, not only its final pose.
- Use the same sign variant shown in the reference dataset videos.
- Move naturally during the 48-frame recording.
- Try again if the application reports an uncertain result.

## Troubleshooting

### `uv` is not recognised

Install `uv` using the command in the setup section, close PowerShell, and open it again.

### Required model not found

Update the repository:

```powershell
git switch main
git pull origin main
```

Then verify the model files again:

```powershell
Test-Path .\models\hand_landmarker.task
Test-Path .\models\gesture_sequence_adapted_v3.joblib
```

### Webcam cannot be opened

Close applications that may already be using the webcam, such as:

- Microsoft Teams
- Google Meet
- Zoom
- Windows Camera

Then run:

```powershell
uv run python .\app.py
```

Also verify Windows camera permissions:

```text
Settings → Privacy & security → Camera
```

### The prediction is uncertain

- Improve the lighting.
- Keep both hands fully visible.
- Move slightly farther away from the camera.
- Perform the complete gesture within the recording period.
- Repeat the sign using the reference variant.

### No speech is heard

- Check the Windows volume.
- Confirm the correct speaker is selected.
- Press `S` to speak the result manually.
- Restart the application if the Windows speech service is unavailable.

## Model Evaluation

The emergency sequence model was developed using:

- Emergency ISL videos from multiple participants
- MediaPipe hand landmarks
- 48-frame motion sequences
- Signer-disjoint evaluation
- Additional webcam adaptation samples

The adapted model achieved **97.44% accuracy** and **0.9747 Macro F1** on the untouched external test set.

These numbers represent performance on the available research dataset. Real-world performance can still be affected by lighting, camera position, sign variation, occlusion and signer differences.

## Important Limitations

- This is a research and hackathon prototype.
- It recognises only the eight supported emergency signs.
- It does not understand continuous ISL sentences.
- It does not replace a qualified ISL interpreter.
- Regional and individual sign variations may affect predictions.
- Unsupported signs should not be treated as valid emergency translations.
- Further testing with more Deaf ISL users and sign-language experts is required.

## Privacy and Offline Operation

After the dependencies and model files have been installed, the recognition and speech pipeline runs locally.

The stable application does not require:

- A cloud AI API
- An internet connection during recognition
- Wearable sensors
- Special gloves
- External tracking hardware

Webcam frames are processed locally by the application.

## Stop the Application

Press:

```text
Q
```

If the application window is unresponsive, return to the PowerShell terminal and press:

```text
Ctrl+C
```