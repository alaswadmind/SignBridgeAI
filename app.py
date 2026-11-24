"""Local SignBridge web app: browser video -> MediaPipe 89 features -> MLP."""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from signbridge_model import BUNDLE, SignBridgePredictor

ROOT = Path(__file__).resolve().parent
WORKER_SCRIPT = ROOT / "mediapipe_worker.py"
MAX_FRAME_BYTES = 1_500_000

app = Flask(__name__, template_folder="Templates", static_folder="web_static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 2_200_000
predictor = SignBridgePredictor(BUNDLE)


class MediaPipeWorker:
    """One persistent Python 3.11 MediaPipe process; serialized for Flask threads."""

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def _command(self) -> list[str]:
        configured = os.environ.get("SIGNBRIDGE_MEDIAPIPE_PYTHON")
        if configured:
            return [configured, "-u", str(WORKER_SCRIPT)]
        current = subprocess.run(
            [sys.executable, "-c", "import mediapipe"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if current.returncode == 0:
            return [sys.executable, "-u", str(WORKER_SCRIPT)]
        launcher = shutil.which("py")
        if launcher:
            return [launcher, "-3.11", "-u", str(WORKER_SCRIPT)]
        raise RuntimeError(
            "MediaPipe is not installed for this Python. Install a Python 3.11 "
            "environment with mediapipe==0.10.21, or set SIGNBRIDGE_MEDIAPIPE_PYTHON."
        )

    def _start(self) -> None:
        command = self._command()
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=flags,
        )

    def extract(self, jpeg_b64: str) -> dict:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self._start()
            assert self.process is not None and self.process.stdin is not None and self.process.stdout is not None
            self.process.stdin.write(json.dumps({"jpeg": jpeg_b64}) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                code = self.process.poll()
                self.process = None
                raise RuntimeError(f"MediaPipe worker stopped unexpectedly (exit={code})")
            response = json.loads(line)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "MediaPipe feature extraction failed"))
            return response

    def close(self) -> None:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.process = None


worker = MediaPipeWorker()


@app.get("/")
def home():
    classes = [
        {"id": int(label), "arabic": predictor.label_map[str(int(label))]["arabic"],
         "name": predictor.label_map[str(int(label))]["name"]}
        for label in predictor.encoder.classes_
    ]
    return render_template("signbridge.html", classes=classes, feature_cols=predictor.feature_cols)


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ready",
        "model": "SignMLP",
        "features": predictor.in_features,
        "classes": predictor.num_classes,
        "feature_order": "63 MediaPipe XYZ coordinates + 26 validated angles",
    })


@app.post("/api/predict")
def predict_frame():
    body = request.get_json(silent=True) or {}
    image_data = body.get("image", "")
    if not isinstance(image_data, str) or not image_data:
        return jsonify({"ok": False, "error": "لم تصل صورة من الكاميرا."}), 400
    if image_data.startswith("data:"):
        image_data = image_data.partition(",")[2]
    if len(image_data) > MAX_FRAME_BYTES * 1.5:
        return jsonify({"ok": False, "error": "حجم إطار الكاميرا أكبر من الحد المسموح."}), 413
    try:
        raw = base64.b64decode(image_data, validate=True)
        if len(raw) > MAX_FRAME_BYTES:
            return jsonify({"ok": False, "error": "حجم إطار الكاميرا أكبر من الحد المسموح."}), 413
        extracted = worker.extract(image_data)
        if not extracted.get("hand_detected"):
            return jsonify({"ok": True, "hand_detected": False})
        prediction = predictor.predict(extracted["features"])
        return jsonify({
            "ok": True,
            "hand_detected": True,
            "hand_count": extracted["hand_count"],
            "landmarks": extracted["landmarks"],
            "features": extracted["features"],
            "feature_count": len(extracted["features"]),
            "image_width": extracted["image_width"],
            "image_height": extracted["image_height"],
            **prediction,
        })
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        app.logger.warning("Frame inference failed: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 503


if __name__ == "__main__":
    try:
        app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False, threaded=True)
    finally:
        worker.close()
