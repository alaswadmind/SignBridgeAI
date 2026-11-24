# SignBridge AI

A local web prototype for recognizing a set of Arabic sign-language hand signs from a webcam frame. The trained model consumes the same 89 coordinate and angle features used during training; it does not classify image pixels directly.

## Pipeline

```text
Browser camera frame → MediaPipe Hands (21 landmarks)
→ 63 XYZ coordinates + 26 hand angles (89 features)
→ training StandardScaler → coordinate-based SignMLP (32 classes)
→ Arabic label and editable text area
```

The current prototype recognizes isolated classes. It does not translate sentences or infer full linguistic context. Natural Language Processing and contextual interpretation are future research and development directions.

## Run locally on Windows

The web server needs Python, Flask, OpenCV, NumPy, scikit-learn, and PyTorch. The MediaPipe worker is launched as a separate process and is tested with Python 3.11.

```powershell
python -m pip install -r requirements-web.txt
py -3.11 -m pip install -r requirements-mediapipe-worker.txt
python .\app.py
```

Open <http://127.0.0.1:5000>, choose **تشغيل الكاميرا**, and allow camera access. Keep the PowerShell window open while using the app. If PowerShell blocks `.ps1` files, running `python .\app.py` avoids the execution-policy issue.

Set `SIGNBRIDGE_MEDIAPIPE_PYTHON` to a Python executable path if the `py -3.11` launcher is unavailable. The camera is accessed by the browser; remote devices require HTTPS.

## Model files

`signbridge_model/` contains the supplied MLP checkpoint, fitted scaler, label encoder, Arabic label map, and training notebook. The checkpoint feature order is validated at startup. The pickled scaler and encoder were created with scikit-learn 1.6.1; using another version may emit a compatibility warning.

## Development notes

- `aslad_feature_extraction.py` contains the reusable coordinate and angle extraction pipeline.
- `mediapipe_worker.py` keeps MediaPipe loaded in a persistent worker process.
- `ASLAD_REVERSE_ENGINEERING_REPORT.md` documents the feature reconstruction and its validation limits.
- `exports/` contains the short promotional video, SRT captions, and suggested Arabic voice-over text.
- The Flask development server is intended for local prototyping, not production deployment.
