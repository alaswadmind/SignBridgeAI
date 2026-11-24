"""Compare MediaPipe Hands on the ten supplied JPGs with the reference CSV."""
import csv
import json
from pathlib import Path

import numpy as np

from aslad_reverse_engineer import ANGLE_TRIPLETS, ROOT, SAMPLE, read_reference, angle

records = read_reference()
data = json.loads((ROOT / "aslad_analysis.json").read_text(encoding="utf-8"))
mp_data = data["mediapipe"]
if not mp_data.get("available"):
    raise SystemExit(f"MediaPipe unavailable: {mp_data.get('error')}")

summary_rows = []
coord_rows = []
angle_rows = []
feature_errors = []
all_coord_errors, all_angle_errors = [], []
VARIANTS = ("bgr2rgb", "raw_bgr", "rgb_2x", "rgb_4x", "rgb_flip", "rgb_flip_unmirrored")
for image, ref in records.items():
    result_variants = mp_data["results"][image]
    for variant in VARIANTS:
        hands = result_variants.get(variant)
        if not hands:
            summary_rows.append([image, variant, 0, "", "", "", "", "", "", "", "", ""])
            continue
        candidates = [(np.asarray(h, dtype=float), i) for i, h in enumerate(hands)]
        # Retrospective nearest-hand match is used only to evaluate detections, not in the extractor.
        points, hand_index = min(candidates, key=lambda z: float(np.mean(np.abs(z[0] - ref["coords"]))))
        coord_diff = np.abs(points - ref["coords"])
        calc_angles = np.array([angle(points[a], points[v], points[c]) for a, v, c in ANGLE_TRIPLETS])
        angle_diff = np.abs(calc_angles - ref["angles"])
        all_coord_errors.extend(coord_diff.ravel().tolist())
        all_angle_errors.extend(angle_diff.tolist())
        feature_errors.extend(coord_diff.ravel().tolist() + angle_diff.tolist())
        summary_rows.append([image, variant, len(hands), hand_index,
                             float(np.mean(coord_diff)), float(np.sqrt(np.mean(coord_diff**2))),
                             float(np.max(coord_diff)), int(np.sum(coord_diff <= 1e-3)),
                             float(np.mean(angle_diff)), float(np.sqrt(np.mean(angle_diff**2))),
                             float(np.max(angle_diff)), int(np.sum(angle_diff <= .1))])
        for i in range(21):
            for dim, axis in enumerate("XYZ"):
                coord_rows.append([image, variant, hand_index, i, axis,
                                   ref["coords"][i, dim], points[i, dim], coord_diff[i, dim]])
        for i, (triplet, target, value, err) in enumerate(zip(ANGLE_TRIPLETS, ref["angles"], calc_angles, angle_diff), 1):
            angle_rows.append([image, variant, i, 4158 + i, f"{triplet[0]}-{triplet[1]}-{triplet[2]}", target, value, err])

for filename, header, rows in [
    ("aslad_landmark_coordinate_comparison.csv", ["image", "input_variant", "selected_hand", "landmark", "axis", "reference", "mediapipe", "abs_error"], coord_rows),
    ("aslad_angle_comparison.csv", ["image", "input_variant", "angle_number", "csv_column", "points_a_vertex_c", "reference_degrees", "mediapipe_degrees", "abs_error_degrees"], angle_rows),
    ("aslad_per_image_comparison.csv", ["image", "input_variant", "detected_hands", "selected_hand", "coordinate_MAE", "coordinate_RMSE", "coordinate_max_abs_error", "coords_within_1e-3_of_63", "angle_MAE_degrees", "angle_RMSE_degrees", "angle_max_abs_error_degrees", "angles_within_0.1deg_of_26"], summary_rows),
]:
    with (ROOT / filename).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def metrics(x):
    a = np.asarray(x, dtype=float)
    return {"mae": float(np.mean(np.abs(a))), "rmse": float(np.sqrt(np.mean(a*a))), "max_abs": float(np.max(np.abs(a)))}

summary = {
    "mediapipe_version": mp_data["version"],
    "note": "RGB and raw BGR are tested. Images are fed at original size. For each detected result, nearest hand to reference is selected retrospectively for evaluation only.",
    "variants": {},
}
for variant in VARIANTS:
    eligible = [r for r in summary_rows if r[1] == variant and r[2]]
    ce, ae = [], []
    for image in records:
        hands = mp_data["results"][image].get(variant)
        if hands:
            ref = records[image]["coords"]
            pts = min((np.asarray(h, float) for h in hands), key=lambda p: float(np.mean(np.abs(p-ref))))
            ce.extend((pts-ref).ravel().tolist())
            aa = np.array([angle(pts[a], pts[v], pts[c]) for a,v,c in ANGLE_TRIPLETS])
            ae.extend((aa-records[image]["angles"]).tolist())
    summary["variants"][variant] = {"detected_images": sum(bool(mp_data["results"][n].get(variant)) for n in records),
                                    "coordinate_errors": metrics(ce) if ce else None,
                                    "angle_errors_degrees": metrics(ae) if ae else None,
                                    "coords_within_1e-3": int(np.sum(np.abs(ce) <= 1e-3)),
                                    "angles_within_0.1deg": int(np.sum(np.abs(ae) <= .1))}
(ROOT / "aslad_image_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
