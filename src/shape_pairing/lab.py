"""从绘制参数生成本地拼板。每题上下两排的原始图块都保留像素尺寸。"""
import math

import numpy as np
from PIL import Image, ImageDraw

from lab.choices import bank
from lab.helpers import _png

TILE = 96
POLYGONS = {
    "tri": [(0, -1), (1, 1), (-1, 1)],
    "trapezoid": [(-.72, -1), (.72, -1), (1, 1), (-1, 1)],
    "plus": [(-.32, -1), (.32, -1), (.32, -.32), (1, -.32), (1, .32), (.32, .32),
             (.32, 1), (-.32, 1), (-.32, .32), (-1, .32), (-1, -.32), (-.32, -.32)],
    "s": [(-1, 0), (-1/3, 0), (-1/3, -1), (1, -1), (1, 0), (1/3, 0), (1/3, 1), (-1, 1)],
}


def _tile(row, pose):
    shape = row["shape"]
    width, height = int(row["width"]), int(row["height"])
    angle, dx, dy = map(int, row[pose].split(","))
    side = TILE * 4
    image = Image.new("L", (side, side))
    draw = ImageDraw.Draw(image)

    def xy(x, y):
        return round(side / 2 + x * width * 2), round(side / 2 + y * height * 2)

    bounds = (xy(-1, -1), xy(1, 1))
    if shape in {"rect", "round-window", "square-window", "offset-window"}:
        draw.rectangle(bounds, fill=255)
        if shape == "square-window":
            draw.rectangle((xy(-.30, -.30), xy(.30, .30)), fill=0)
        elif shape.endswith("window"):
            shift = .36 if shape == "offset-window" else 0
            draw.ellipse((xy(shift - .34, -.34), xy(shift + .34, .34)), fill=0)
    elif shape in {"ellipse", "ring"}:
        draw.ellipse(bounds, fill=255)
        if shape == "ring":
            draw.ellipse((xy(-.48, -.48), xy(.48, .48)), fill=0)
    else:
        if shape in {"pent", "hex", "oct", "star"}:
            count = {"pent": 5, "hex": 6, "oct": 8, "star": 10}[shape]
            points = []
            for i in range(count):
                phi = 2 * math.pi * i / count - math.pi / 2
                radius = .42 if shape == "star" and i % 2 else 1
                points.append((radius * math.cos(phi), radius * math.sin(phi)))
        elif shape == "z":
            points = [(-x, y) for x, y in POLYGONS["s"]]
        else:
            points = POLYGONS[shape]
        draw.polygon([xy(x, y) for x, y in points], fill=255)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, translate=(4 * dx, 4 * dy))
    pixels = np.asarray(image.resize((TILE, TILE), Image.Resampling.LANCZOS))
    return np.uint8(pixels >= 128) * 255


def masks(rows):
    pieces = [_tile(row, "piece_pose") for row in rows]
    slots = [_tile(row, "slot_pose") for row in sorted(rows, key=lambda row: int(row["slot"]))]
    return pieces, slots


def generate(rng):
    rows = rng.choice(tuple(bank("shape-pairing").values()))
    pieces, slots = masks(rows)
    public = {"kind": "shape-pairing", "width": TILE * len(rows), "height": (TILE + 22) * 2,
              "instruction": "将每块图形放到对应槽位。可以旋转，尺寸和正反面须一致。也可依次点选拼块和槽位。",
              "pieces": [_png(255 - m) for m in pieces], "slots": [_png(255 - m) for m in slots]}
    return public, {"text": "".join(row["slot"] for row in rows), "ignore_case": False}
