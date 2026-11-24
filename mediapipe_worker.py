"""Persistent line-protocol worker for the Python with a MediaPipe wheel.

Input: one JSON object per line: {"jpeg": "<base64>"}
Output: one JSON object per line with 21 landmarks and the 89 training features.
"""
from __future__ import annotations

import base64
import json
import sys

import cv2
import mediapipe as mp
import numpy as np

from aslad_feature_extraction import extract_from_bgr


def main() -> int:
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.1,
    ) as detector:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                jpeg = base64.b64decode(request["jpeg"], validate=True)
                encoded = np.frombuffer(jpeg, dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                result = extract_from_bgr(image, detector) if image is not None else None
                if result is None:
                    response = {"ok": True, "hand_detected": False}
                else:
                    landmarks, features, hand_count = result
                    if not np.isfinite(features).all():
                        response = {"ok": False, "error": "non_finite_feature"}
                    else:
                        response = {
                            "ok": True,
                            "hand_detected": True,
                            "hand_count": hand_count,
                            "landmarks": landmarks.tolist(),
                            "features": features,
                            "image_width": int(image.shape[1]),
                            "image_height": int(image.shape[0]),
                        }
            except Exception as exc:
                response = {"ok": False, "error": str(exc)}
            sys.stdout.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
