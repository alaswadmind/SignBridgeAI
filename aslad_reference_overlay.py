from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from aslad_reverse_engineer import SAMPLE, read_reference

records = read_reference()
tiles = []
for name, rec in records.items():
    im = cv2.imread(str(SAMPLE / name))
    h, w = im.shape[:2]
    pts = rec["coords"]
    for i, (x, y, z) in enumerate(pts):
        px, py = round(x * w), round(y * h)
        cv2.circle(im, (px, py), 2, (0, 0, 255), -1)
        cv2.putText(im, str(i), (px + 2, py - 2), cv2.FONT_HERSHEY_SIMPLEX, .25, (255, 0, 0), 1)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    tile = Image.fromarray(rgb).resize((396, 266))
    canvas = Image.new("RGB", (396, 290), "white")
    canvas.paste(tile, (0, 24))
    ImageDraw.Draw(canvas).text((5, 4), name, fill="black")
    tiles.append(canvas)
sheet = Image.new("RGB", (396 * 2, 290 * 5), "white")
for i, tile in enumerate(tiles):
    sheet.paste(tile, ((i % 2) * 396, (i // 2) * 290))
sheet.save(Path(__file__).parent / "aslad_reference_landmarks.png")
