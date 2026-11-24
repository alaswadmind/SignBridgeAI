"""Numerical inspection helpers for the supplied ASLAD sample."""
from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).parent
SAMPLE = ROOT / "aslad_sample"
CSV_PATH = SAMPLE / "sample_coordinates.csv"
ANGLE_TRIPLETS = (
    (0, 1, 5), (0, 1, 17), (0, 1, 2), (0, 5, 17), (0, 5, 6), (0, 5, 9),
    (0, 17, 13), (0, 17, 18), (1, 2, 3), (2, 3, 4), (5, 6, 9), (5, 6, 7),
    (5, 9, 10), (5, 9, 13), (6, 7, 8), (9, 10, 13), (9, 10, 11), (9, 13, 14),
    (9, 13, 17), (10, 11, 12), (13, 14, 17), (13, 14, 15), (13, 17, 18),
    (14, 15, 16), (17, 18, 19), (18, 19, 20),
)


def read_reference():
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    records = {}
    for row in rows:
        coords = np.array([float(row[str(i)]) for i in range(4096, 4159)]).reshape(21, 3)
        angles = np.array([float(row[str(i)]) for i in range(4159, 4185)])
        records[row["4191"]] = {"coords": coords, "angles": angles, "label": row["4190"]}
    return records


def image_metadata():
    out = {}
    for p in sorted(SAMPLE.glob("*.jpg")):
        with Image.open(p) as im:
            out[p.name] = {"size": list(im.size), "mode": im.mode, "exif_orientation": im.getexif().get(274, 1)}
    return out


def angle(a, b, c):
    u, v = a - b, c - b
    den = np.linalg.norm(u) * np.linalg.norm(v)
    if den < 1e-12:
        return np.nan
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / den, -1, 1))))


def angle_candidates(records):
    """Search all point-triplet internal angles in 2D and 3D against each CSV angle."""
    names = list(records)
    refs = np.stack([records[n]["angles"] for n in names])
    results = []
    for dims, dim_name in [((0, 1), "XY"), ((0, 1, 2), "XYZ"), ((0, 2), "XZ"), ((1, 2), "YZ")]:
        values = {}
        for i, j, k in itertools.combinations(range(21), 3):
            # each possible vertex of the unordered triple
            for center, (a, c) in [(i, (j, k)), (j, (i, k)), (k, (i, j))]:
                seq = []
                for n in names:
                    pts = records[n]["coords"][:, dims]
                    seq.append(angle(pts[a], pts[center], pts[c]))
                values[(a, center, c)] = np.asarray(seq)
        for col in range(26):
            y = refs[:, col]
            scores = []
            for key, x in values.items():
                d = x - y
                scores.append((float(np.nanmean(np.abs(d))), float(np.nanmax(np.abs(d))), key))
            scores.sort()
            best = scores[:3]
            results.append({"column": 4159 + col, "dimensions": dim_name, "best": best})
    return results


def extract_mediapipe(records):
    try:
        import mediapipe as mp
    except Exception as e:
        return {"available": False, "error": repr(e)}
    outputs = {}
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                  model_complexity=1, min_detection_confidence=0.1) as hands:
        for name in records:
            path = SAMPLE / name
            image = cv2.imread(str(path))
            h, w = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            variants = {
                "bgr2rgb": rgb,
                "raw_bgr": image,
                "rgb_2x": cv2.resize(rgb, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC),
                "rgb_4x": cv2.resize(rgb, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC),
                "rgb_flip": cv2.flip(rgb, 1),
            }
            variant_out = {}
            for vname, rgb in variants.items():
                result = hands.process(rgb)
                if not result.multi_hand_landmarks:
                    variant_out[vname] = None
                    continue
                hands_out = []
                for hand_lms in result.multi_hand_landmarks:
                    # Raw normalized (x,y,z) matches the CSV's apparent coordinate convention.
                    hands_out.append(np.array([[p.x, p.y, p.z] for p in hand_lms.landmark]))
                variant_out[vname] = [x.tolist() for x in hands_out]
                if vname == "rgb_flip" and hands_out:
                    variant_out["rgb_flip_unmirrored"] = [
                        np.column_stack((1.0 - h[:, 0], h[:, 1], h[:, 2])).tolist() for h in hands_out
                    ]
            outputs[name] = variant_out
    return {"available": True, "version": mp.__version__, "results": outputs}


def main():
    records = read_reference()
    stats = {"n_rows": len(records), "images": image_metadata(),
             "coord_min": np.stack([r["coords"] for r in records.values()]).min(axis=(0, 1)).tolist(),
             "coord_max": np.stack([r["coords"] for r in records.values()]).max(axis=(0, 1)).tolist(),
             "angle_min": np.stack([r["angles"] for r in records.values()]).min(axis=0).tolist(),
             "angle_max": np.stack([r["angles"] for r in records.values()]).max(axis=0).tolist(),
             "angle_search": angle_candidates(records),
             "mediapipe": extract_mediapipe(records)}
    (ROOT / "aslad_analysis.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "angle_search"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
