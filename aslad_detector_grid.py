"""Small sensitivity grid for MediaPipe detector settings on the sample."""
import csv, json
from pathlib import Path
import cv2, mediapipe as mp, numpy as np
from aslad_reverse_engineer import ANGLE_TRIPLETS, ROOT, SAMPLE, read_reference, angle

records = read_reference()
grid = [(c, m) for c in (.1, .5, .8) for m in (0, 1)]
summary = {}
for confidence, complexity in grid:
    coords_err, angle_err = [], []
    detected = 0
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
            model_complexity=complexity, min_detection_confidence=confidence) as detector:
        for name, ref in records.items():
            image = cv2.imread(str(SAMPLE / name))
            result = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not result.multi_hand_landmarks:
                continue
            detected += 1
            candidates = [np.array([[p.x, p.y, p.z] for p in h.landmark]) for h in result.multi_hand_landmarks]
            points = min(candidates, key=lambda x: float(np.mean(np.abs(x-ref["coords"]))))
            coords_err.extend((points-ref["coords"]).ravel().tolist())
            angles = np.array([angle(points[a], points[v], points[c]) for a,v,c in ANGLE_TRIPLETS])
            angle_err.extend((angles-ref["angles"]).tolist())
    c, a = np.abs(coords_err), np.abs(angle_err)
    summary[f"confidence={confidence},complexity={complexity}"] = {
        "detected_images": detected,
        "coordinate_MAE": float(c.mean()) if len(c) else None,
        "coordinate_RMSE": float(np.sqrt(np.mean(c*c))) if len(c) else None,
        "coordinate_max_abs": float(c.max()) if len(c) else None,
        "coordinates_within_1e-3": int(np.sum(c <= 1e-3)),
        "angle_MAE_degrees": float(a.mean()) if len(a) else None,
        "angle_RMSE_degrees": float(np.sqrt(np.mean(a*a))) if len(a) else None,
        "angle_max_abs_degrees": float(a.max()) if len(a) else None,
        "angles_within_0.1deg": int(np.sum(a <= .1)),
    }
(ROOT / "aslad_detector_grid.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
