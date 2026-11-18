"""按印刷颜色和原始字高识别固定字卡。"""
from functools import cache

import cv2
import numpy as np

from .cards import REFERENCE_FONTS, glyph_png

# 四种参考墨色在 OpenCV 0–179 色相范围内的值。
HUES = {"红": 1, "绿": 70, "蓝": 109, "橙": 17}


def centered_glyph(mask):
    x, y, w, h = cv2.boundingRect(mask)
    left, top = (mask.shape[1] - w) // 2, (mask.shape[0] - h) // 2
    centered = np.zeros_like(mask, dtype=bool)
    centered[top:top+h, left:left+w] = mask[y:y+h, x:x+w] != 0
    return centered


@cache
def templates():
    refs = []
    for letter in "TtCcPpAa":
        for font in REFERENCE_FONTS:
            data = glyph_png(letter, font, 55, 0, "#141414", mask=True)
            mask = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
            refs.append((letter, centered_glyph(mask)))
    return refs


def read(png, _question):
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv[:, :, 1], 45, 255)
    if not mask.any():
        return "", None
    hue = float(np.median(hsv[:, :, 0][mask > 0]))
    color = min(HUES, key=lambda c: min(abs(hue - HUES[c]), 180 - abs(hue - HUES[c])))
    glyph = centered_glyph(mask)
    letter, _ = max(templates(), key=lambda ref: np.count_nonzero(glyph & ref[1]) / np.count_nonzero(glyph | ref[1]))
    label = color + letter
    return label, label
