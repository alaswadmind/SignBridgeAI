# ASLAD-190K feature reverse engineering

## Status

This report is based on the supplied ten-image sample and its 91-column CSV. The 26 angle equations are recovered with numerical equality over all ten rows. JPG inference was run using MediaPipe 0.10.21 and OpenCV 4.11.0 on Python 3.11; it finds hands in all ten images, but its coordinates do not reproduce the CSV exactly. The runnable candidate extractor is [aslad_feature_extraction.py](aslad_feature_extraction.py); the exploratory analysis is [aslad_reverse_engineer.py](aslad_reverse_engineer.py).

## Dataset findings

- The provided CSV has 11 rows including a header, 91 columns per row, and 10 image records. Header names are `4096`–`4184`, `4190` and `4191`; it is a sparse excerpt of a wider original feature table, not a 4,192-column CSV.
- The images are RGB JPGs with EXIF orientation 1. Resolutions range from 195×125 to 327×166 pixels. No EXIF rotation is indicated.
- Across the ten records, X ranges 0.145522–0.871537 and Y 0.206897–0.784483. Z ranges −0.254408–0.056940. Thus X/Y are normalized image coordinates, while Z is a relative depth coordinate rather than image pixels or a metric distance.
- X/Y lie within [0,1], which does not indicate a crop or horizontal flip. These ten values alone cannot prove whether the source performed a crop/resize that happened to preserve the shown image dimensions.
- The 63 values form 21 interleaved `(x,y,z)` triplets. The names/order below follow the MediaPipe Hands 21-landmark convention; they are a strong interpretation of the feature shape, but point-by-point JPG reproduction remains unverified.
- The 26 angles range from 7.337° to 177.753°, so they are degrees rather than radians.

## Evidence from the attached dataset paper

The newly supplied paper text narrows the likely tooling, but does not specify the actual feature-extraction code:

- It says CVZone, OpenCV and MediaPipe were used to capture RGB images of varied sizes. This is direct evidence for the **image capture** stack.
- It says the dataset includes images where MediaPipe detects hands and images where it does not. This supports hand detection being part of the data workflow, but it does not say whether undetected images were kept in or excluded from the CSV.
- For feature generation, it says a keypoint model “such as MediaPipe” was used and that 26 angles were extracted. It gives no exact version, parameters, point triplets, image transform, or angle code in the supplied text.
- The text alternates between a balanced 96,000-image set and “90,000” images for feature extraction. Those counts conflict internally; neither changes what the ten supplied rows show.

I also tested CVZone 1.6.1, the legacy wrapper matching the paper's cited CVZone/MediaPipe era. Its implementation converts BGR to RGB and exposes integer pixel landmarks. Converting those pixels back to normalized XYZ does not reproduce the CSV. With `staticMode=False` on this batch, raw normalized landmarks gave MAE 0.03979 and angle MAE 5.0411°; with `staticMode=True`, it matched the direct MediaPipe run (coordinate MAE 0.04780; angle MAE 4.8459°). The pixel-rounded output was slightly closer in coordinate MAE (0.03964) but still not exact. This strengthens the hypothesis that MediaPipe Hands was involved, but does not identify the original version or settings. CVZone 2.0.0 was not inference-tested because it requires downloading its separate Hand Landmarker model and that download was refused by the execution environment.

## Landmark order

| Index | Landmark | Index | Landmark | Index | Landmark |
|---:|---|---:|---|---:|---|
| 0 | Wrist | 1 | Thumb CMC | 2 | Thumb MCP |
| 3 | Thumb IP | 4 | Thumb tip | 5 | Index MCP |
| 6 | Index PIP | 7 | Index DIP | 8 | Index tip |
| 9 | Middle MCP | 10 | Middle PIP | 11 | Middle DIP |
| 12 | Middle tip | 13 | Ring MCP | 14 | Ring PIP |
| 15 | Ring DIP | 16 | Ring tip | 17 | Pinky MCP |
| 18 | Pinky PIP | 19 | Pinky DIP | 20 | Pinky tip |

## Recovered angle definitions

Every column is the 3D internal angle `acos(dot(Pa-Pv, Pc-Pv) / (norm(Pa-Pv) * norm(Pc-Pv)))`, in degrees. The bracket lists `(point A, vertex, point C)` using the table above. For each column, the best fit across all ten records has mean absolute error below `9e-15°` and maximum error below `3e-14°`, consistent with floating point roundoff.

| Angle | CSV column | Points `(A, vertex, C)` | Formula |
|---:|---:|---|---|
| 1 | 4159 | (0, 1, 5) | 3D internal angle |
| 2 | 4160 | (0, 1, 17) | 3D internal angle |
| 3 | 4161 | (0, 1, 2) | 3D internal angle |
| 4 | 4162 | (0, 5, 17) | 3D internal angle |
| 5 | 4163 | (0, 5, 6) | 3D internal angle |
| 6 | 4164 | (0, 5, 9) | 3D internal angle |
| 7 | 4165 | (0, 17, 13) | 3D internal angle |
| 8 | 4166 | (0, 17, 18) | 3D internal angle |
| 9 | 4167 | (1, 2, 3) | 3D internal angle |
| 10 | 4168 | (2, 3, 4) | 3D internal angle |
| 11 | 4169 | (5, 6, 9) | 3D internal angle |
| 12 | 4170 | (5, 6, 7) | 3D internal angle |
| 13 | 4171 | (5, 9, 10) | 3D internal angle |
| 14 | 4172 | (5, 9, 13) | 3D internal angle |
| 15 | 4173 | (6, 7, 8) | 3D internal angle |
| 16 | 4174 | (9, 10, 13) | 3D internal angle |
| 17 | 4175 | (9, 10, 11) | 3D internal angle |
| 18 | 4176 | (9, 13, 14) | 3D internal angle |
| 19 | 4177 | (9, 13, 17) | 3D internal angle |
| 20 | 4178 | (10, 11, 12) | 3D internal angle |
| 21 | 4179 | (13, 14, 17) | 3D internal angle |
| 22 | 4180 | (13, 14, 15) | 3D internal angle |
| 23 | 4181 | (13, 17, 18) | 3D internal angle |
| 24 | 4182 | (14, 15, 16) | 3D internal angle |
| 25 | 4183 | (17, 18, 19) | 3D internal angle |
| 26 | 4184 | (18, 19, 20) | 3D internal angle |

These equations were obtained by exhaustive search of internal angles over point triplets using the supplied coordinates, separately for XY, XYZ, XZ and YZ. Only XYZ produced all 26 columns with machine-precision agreement over every row. This confirms the equations for the supplied CSV, not that the source authors used the same implementation or library version.

## Comparison and completion limits

| Check | Result |
|---|---|
| CSV rows linked to JPG names | 10/10 |
| 21×3 coordinate vector shape | 10/10 |
| Angle formulas discovered | 26/26 |
| Angle values regenerated from reference coordinates | 260/260 equal within 1e-10° |
| JPG → MediaPipe hand detection | 10/10 with RGB, static image mode, confidence 0.1, model complexity 1 |
| Best direct RGB 63-coordinate comparison | MAE 0.047805, RMSE 0.057613, maximum absolute error 0.149953; 14/630 coordinates within 0.001 |
| Direct RGB 26-angle comparison | MAE 4.8459°, RMSE 8.2661°, maximum absolute error 72.6643°; 7/260 angles within 0.1° |
| Best-tested image variant | Horizontally flipped then mapped back: coordinate MAE 0.04110 over 9/10 detections, but angle MAE 5.3852°; it does not recover the original pipeline |
| Raw BGR test | 0/10 detections |
| Full 89-feature image comparison | Measured, but nowhere near exact: 21/890 (2.36%) features meet the selected per-feature tolerances (0.001 coordinate, 0.1° angle) for the direct RGB run |

The coordinate-to-angle check is exact to floating point, while image-to-coordinate comparison is not. Additional tested variants were RGB resized 2× and 4× and horizontal mirroring. The direct RGB/original-size path gave the best mean angle error among full-coverage settings. A sensitivity grid over detection confidence 0.1, 0.5 and 0.8 and model complexity 0 and 1 did not find a setting that reconciled the outputs. A higher-confidence setting reduced angle error to 3.68° but detected only 9/10. This indicates a model/version/preprocessing mismatch or loss in the supplied low-resolution JPGs; the exact image extraction model/version and any source resizing/cropping are not recoverable from the sample alone. The final extractor is therefore a practical compatible-format candidate, not a proven faithful reimplementation.

| Image | Hand detected | Coordinate MAE | Angle MAE (°) |
|---|---:|---:|---:|
| ain (5191).jpg | 1 | 0.04991 | 4.72 |
| ain (3810).jpg | 1 | 0.06976 | 14.44 |
| ain (4916).jpg | 1 | 0.03673 | 3.94 |
| ain (3471).jpg | 1 | 0.04428 | 2.38 |
| ain (4095).jpg | 2 | 0.04712 | 2.88 |
| ain (5217).jpg | 1 | 0.04989 | 2.22 |
| ain (4954).jpg | 1 | 0.03954 | 3.77 |
| ain (5102).jpg | 1 | 0.04281 | 4.44 |
| ain (3562).jpg | 1 | 0.05356 | 4.85 |
| ain (4705).jpg | 1 | 0.04444 | 4.83 |

The generated [experimental CSV](aslad_experimental_output.csv) contains actual MediaPipe output for all ten JPGs. The detailed per-image and per-coordinate comparisons are in [aslad_experimental_csv_comparison.csv](aslad_experimental_csv_comparison.csv), [aslad_landmark_coordinate_comparison.csv](aslad_landmark_coordinate_comparison.csv), and [aslad_angle_comparison.csv](aslad_angle_comparison.csv). The CVZone run results are in [aslad_cvzone_comparison.json](aslad_cvzone_comparison.json). [aslad_reference_rebuilt.csv](aslad_reference_rebuilt.csv) is explicitly a separate coordinate-to-angle check: it reuses reference coordinates and is not an image-inference output.

Run the candidate on a folder with `python aslad_feature_extraction.py path/to/images --output features.csv`. The tested dependency versions are listed in [requirements-aslad.txt](requirements-aslad.txt). Python 3.11 was used for this run; MediaPipe 0.10.21 has no wheel for the active Python 3.14 interpreter.

## Sources

- The paper identifies the model and reports 96,000 collected images with hand-landmark angles; its public page does not expose the extraction code: [Springer article](https://link.springer.com/article/10.1007/s00521-024-10729-7).
- Dataset authors describe 190,000 RGB images captured with two signers/webcams and state MediaPipe was used in collection: [Mendeley Data record](https://data.mendeley.com/datasets/2fgpn5dwgc/3).
- The official MediaPipe documentation says Hand Landmarker outputs image-coordinate landmarks, world-coordinate landmarks and handedness for one or more hands: [Google AI Edge Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).
- No author source code was found in the project workspace or in the cited public paper/dataset records during this check.
