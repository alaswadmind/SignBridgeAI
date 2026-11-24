"""Compare the actual extractor CSV with the ten reference rows."""
import csv, json
from pathlib import Path
import numpy as np
from aslad_reverse_engineer import ROOT, read_reference

records = read_reference()
out_csv = ROOT / "aslad_experimental_output.csv"
with out_csv.open(encoding="utf-8-sig", newline="") as f:
    produced = {r["4191"]: r for r in csv.DictReader(f)}
rows, coord_errors, angle_errors = [], [], []
for name, ref in records.items():
    if name not in produced:
        rows.append([name, "no detection", "", "", "", "", "", ""])
        continue
    row = produced[name]
    c = np.array([float(row[str(i)]) for i in range(4096,4159)]).reshape(21,3)
    a = np.array([float(row[str(i)]) for i in range(4159,4185)])
    cd, ad = np.abs(c-ref["coords"]), np.abs(a-ref["angles"])
    coord_errors.extend(cd.ravel().tolist()); angle_errors.extend(ad.tolist())
    rows.append([name, "detected", float(cd.mean()), float(np.sqrt(np.mean(cd**2))), float(cd.max()),
                 int((cd <= 1e-3).sum()), float(ad.mean()), float(np.sqrt(np.mean(ad**2))),
                 float(ad.max()), int((ad <= .1).sum())])
with (ROOT / "aslad_experimental_csv_comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f); w.writerow(["image","status","coordinate_MAE","coordinate_RMSE","coordinate_max_abs","coordinates_within_1e-3_of_63","angle_MAE_deg","angle_RMSE_deg","angle_max_abs_deg","angles_within_0.1deg_of_26"]); w.writerows(rows)
c, a = np.asarray(coord_errors), np.asarray(angle_errors)
summary = {
    "mediapipe_version": "0.10.21",
    "rows_detected": len(produced), "rows_expected": len(records),
    "coordinates": {"n": int(c.size), "MAE": float(c.mean()), "RMSE": float(np.sqrt(np.mean(c*c))),
                    "max_abs": float(c.max()), "within_1e-3": int((c<=1e-3).sum())},
    "angles_degrees": {"n": int(a.size), "MAE": float(a.mean()), "RMSE": float(np.sqrt(np.mean(a*a))),
                       "max_abs": float(a.max()), "within_0.1deg": int((a<=.1).sum())},
}
(ROOT / "aslad_experimental_csv_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
