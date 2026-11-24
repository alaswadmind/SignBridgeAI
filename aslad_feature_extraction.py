#!/usr/bin/env python3
"""Extract ASLAD-190K-compatible MediaPipe hand features from images.

Usage:
  python aslad_feature_extraction.py images/ --output features.csv

Coordinates are MediaPipe normalized (x, y, z), in landmark index order.
The 26 validated angles are 3-D internal angles in degrees. The original
sample's source columns are kept as CSV headers: 4096..4158, 4159..4184.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # gives a useful message if user starts with Python 3.14
    raise SystemExit("MediaPipe is required. Install a supported build, e.g. "
                     "python -m pip install mediapipe==0.10.21") from exc

LOG = logging.getLogger("aslad_feature_extraction")
ANGLE_TRIPLETS = (
    (0, 1, 5), (0, 1, 17), (0, 1, 2), (0, 5, 17), (0, 5, 6), (0, 5, 9),
    (0, 17, 13), (0, 17, 18), (1, 2, 3), (2, 3, 4), (5, 6, 9), (5, 6, 7),
    (5, 9, 10), (5, 9, 13), (6, 7, 8), (9, 10, 13), (9, 10, 11), (9, 13, 14),
    (9, 13, 17), (10, 11, 12), (13, 14, 17), (13, 14, 15), (13, 17, 18),
    (14, 15, 16), (17, 18, 19), (18, 19, 20),
)
COORD_HEADERS = [str(i) for i in range(4096, 4159)]
ANGLE_HEADERS = [str(i) for i in range(4159, 4185)]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_angle(points: np.ndarray, triplet: tuple[int, int, int]) -> float:
    a, vertex, c = (points[i] for i in triplet)
    u, v = a - vertex, c - vertex
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom < 1e-12:
        return math.nan
    cosine = float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def extract_from_bgr(image: np.ndarray, detector) -> tuple[np.ndarray, list[float], int] | None:
    """Return selected hand landmarks, the 89-value vector, and hand count."""
    if image is None or image.ndim != 3:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    candidates = []
    for hand in result.multi_hand_landmarks:
        pts = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float64)
        span = np.ptp(pts[:, :2], axis=0)
        candidates.append((float(span[0] * span[1]), pts))
    # One hand per sample; for new images use the most prominent detected hand.
    candidates.sort(key=lambda item: item[0], reverse=True)
    points = candidates[0][1]
    return points, extract_one(points), len(candidates)


def hand_landmarks(image_path: Path, detector) -> np.ndarray | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        LOG.error("Could not read image: %s", image_path)
        return None
    extracted = extract_from_bgr(image, detector)
    if extracted is None:
        LOG.warning("No hand detected: %s", image_path.name)
        return None
    return extracted[0]


def extract_one(points: np.ndarray) -> list[float]:
    return points.reshape(-1).tolist() + [compute_angle(points, t) for t in ANGLE_TRIPLETS]


def image_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="image file or folder of images")
    parser.add_argument("--output", type=Path, default=Path("aslad_features.csv"))
    parser.add_argument("--label", default="", help="optional label; blank by default")
    parser.add_argument("--min-detection-confidence", type=float, default=0.1)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s")
    if not args.input.exists():
        parser.error(f"input does not exist: {args.input}")
    files = image_files(args.input)
    if not files:
        parser.error(f"no supported images found in: {args.input}")
    rows = []
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=args.max_hands,
                                  model_complexity=1,
                                  min_detection_confidence=args.min_detection_confidence) as detector:
        for path in files:
            try:
                points = hand_landmarks(path, detector)
                if points is None:
                    continue
                rows.append(extract_one(points) + [args.label, path.name])
            except Exception:
                LOG.exception("Failed processing %s", path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COORD_HEADERS + ANGLE_HEADERS + ["4190", "4191"])
        writer.writerows(rows)
    LOG.info("Wrote %d/%d rows to %s (MediaPipe %s)", len(rows), len(files), args.output, mp.__version__)
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
