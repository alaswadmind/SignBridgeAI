"""Compare CVZone's wrapper and returned pixel landmarks with reference CSV."""
import json
from pathlib import Path
import cv2, numpy as np
from cvzone.HandTrackingModule import HandDetector
from aslad_reverse_engineer import ANGLE_TRIPLETS, ROOT, SAMPLE, read_reference, angle

records = read_reference()
image_names = sorted(records)
summary = {}
for static in (True, False):
  for confidence in (0.1, 0.5):
    det = HandDetector(staticMode=static, maxHands=2, modelComplexity=1,
                       detectionCon=confidence, minTrackCon=0.5)
    errs_raw, errs_pixels, errs_angles = [], [], []
    detected = 0
    for name in image_names:
      im = cv2.imread(str(SAMPLE/name)); h,w = im.shape[:2]
      found, _ = det.findHands(im.copy(), draw=False, flipType=False)
      if not found:
        continue
      detected += 1
      raw_hands = [np.array([[p.x,p.y,p.z] for p in hand.landmark]) for hand in det.results.multi_hand_landmarks]
      px_hands = [np.array(hand["lmList"],dtype=float) / np.array([w,h,w]) for hand in found]
      ref = records[name]["coords"]
      raw = min(raw_hands,key=lambda x:float(np.mean(np.abs(x-ref))))
      px = min(px_hands,key=lambda x:float(np.mean(np.abs(x-ref))))
      errs_raw.extend(np.abs(raw-ref).ravel().tolist())
      errs_pixels.extend(np.abs(px-ref).ravel().tolist())
      ang = np.array([angle(raw[a],raw[v],raw[c]) for a,v,c in ANGLE_TRIPLETS])
      errs_angles.extend(np.abs(ang-records[name]["angles"]).tolist())
    def m(e, tol=1e-3):
      a=np.asarray(e,float)
      return {"MAE":float(a.mean()),"RMSE":float(np.sqrt(np.mean(a*a))),"max_abs":float(a.max()),f"within_{tol}":int(np.sum(a<=tol))}
    summary[f"staticMode={static},confidence={confidence}"]={"detected_images":detected,"raw_normalized_coords":m(errs_raw),"CVZone_pixel_lmList_normalized":m(errs_pixels),"angles_from_raw_degrees":m(errs_angles,.1)}
(ROOT/"aslad_cvzone_comparison.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
