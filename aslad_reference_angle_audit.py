"""Regenerate the 26 angle columns from supplied reference coordinates."""
import csv
from pathlib import Path

import numpy as np

from aslad_reverse_engineer import ANGLE_TRIPLETS, CSV_PATH, angle, read_reference

ROOT = Path(__file__).parent
records = read_reference()

with (ROOT / "aslad_reference_angle_validation.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "angle_number", "csv_column", "points_a_vertex_c", "reference_degrees",
                "recalculated_degrees", "absolute_error_degrees"])
    for image, rec in records.items():
        pts = rec["coords"]
        for i, (triplet, ref) in enumerate(zip(ANGLE_TRIPLETS, rec["angles"]), 1):
            got = angle(pts[triplet[0]], pts[triplet[1]], pts[triplet[2]])
            w.writerow([image, i, 4158 + i, f"{triplet[0]}-{triplet[1]}-{triplet[2]}",
                        f"{ref:.15g}", f"{got:.15g}", f"{abs(ref-got):.15g}"])

# A reference-coordinate reconstruction, intentionally not presented as JPG inference.
with (ROOT / "aslad_reference_rebuilt.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow([str(i) for i in range(4096, 4185)] + ["4190", "4191"])
    for image, rec in records.items():
        vals = rec["coords"].reshape(-1).tolist()
        vals += [angle(rec["coords"][a], rec["coords"][v], rec["coords"][c])
                 for a, v, c in ANGLE_TRIPLETS]
        w.writerow(vals + [rec["label"], image])

errors = []
for rec in records.values():
    errors.extend(abs(rec["angles"][i] - angle(rec["coords"][a], rec["coords"][v], rec["coords"][c]))
                  for i, (a, v, c) in enumerate(ANGLE_TRIPLETS))
print(f"rows={len(records)}; angles={len(errors)}; MAE={np.mean(errors):.3g}; "
      f"RMSE={np.sqrt(np.mean(np.square(errors))):.3g}; max={np.max(errors):.3g}")
