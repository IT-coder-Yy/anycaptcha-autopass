"""已知提示字的本地 OCR 候选排序；相对分数不是校准后的正确概率。"""
import math
import os

import cv2
import numpy as np
from PIL import Image

from common.vision import best_assignment

MODELS = ("ppocr-v5-mobile.onnx", "ppocr-v6-small.onnx")
ANGLES = (-60, -30, 0, 30, 60)


def prepare(image):
    """遵循 RapidOCR 的 BGR、48 高、保比例、归一化与右侧补零输入。"""
    h, w = image.shape[:2]
    width = min(320, math.ceil(48*w/h))
    tensor = np.zeros((3, 48, 320), np.float32)
    resized = cv2.resize(image, (width, 48)).astype(np.float32).transpose(2, 0, 1)
    tensor[:, :, :width] = (resized/255-.5)/.5
    return tensor


def rank_characters(vision, image, boxes, prompt):
    views = []
    for x1, y1, x2, y2 in boxes:
        crop = Image.fromarray(image[y1:y2, x1:x2, ::-1])
        for angle in ANGLES:
            view = crop.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor="white")
            views.append(prepare(np.asarray(view)[:, :, ::-1]))
    models = []
    for name in MODELS:
        session = vision.session(name)
        chars = ["blank"] + session.get_modelmeta().custom_metadata_map["character"].splitlines() + [" "]
        lookup = {char: index for index, char in enumerate(chars)}
        reads = []
        for start in range(0, len(views), 8):
            raw = session.run(None, {session.get_inputs()[0].name: np.stack(views[start:start+8])})[0]
            if raw.shape[-1] != len(chars) or not np.allclose(raw.sum(axis=-1), 1, atol=.005):
                raise ValueError("OCR 输出与模型字符表或概率格式不一致")
            for probs in raw:
                ids = probs.argmax(axis=-1)
                keep = np.ones(len(ids), bool)
                keep[1:] = ids[1:] != ids[:-1]
                keep &= ids != 0
                reads.append({"text": "".join(chars[int(i)] for i in ids[keep]),
                              "prompt_scores": [float(probs[:, lookup[c]].max()) if c in lookup else 0
                                                for c in prompt]})
        scores = np.array([r["prompt_scores"] for r in reads]).reshape(len(boxes), len(ANGLES), len(prompt)).max(axis=1).T
        assignment, score, gap = best_assignment(scores)
        models.append({"model": name, "scores": scores.tolist(), "assignment": assignment,
                       "score": score, "margin": gap,
                       "unsupported_prompt": list(dict.fromkeys(c for c in prompt if c not in lookup)),
                       "reads": [reads[i:i+len(ANGLES)] for i in range(0, len(reads), len(ANGLES))]})
    return models


def choose_assignment(models, baseline, baseline_usable, margin=.5, floor=1e-6):
    """两个 OCR 整题一致才覆盖旧结果；否则旧结果也须得到至少一个 OCR 支持。"""
    supported = [m for m in models if m["margin"] >= margin and m["score"] >= floor]
    if len(supported) == 2 and supported[0]["assignment"] == supported[1]["assignment"]:
        return supported[0]["assignment"], True, "ppocr_consensus"
    if baseline_usable and any(m["assignment"] == baseline for m in supported):
        return baseline, True, "siamese_ocr_agreement"
    return baseline, False, "insufficient_ocr_support"


def settings():
    margin = float(os.environ.get("CAPTCHA_OCR_MARGIN", ".5"))
    floor = float(os.environ.get("CAPTCHA_OCR_FLOOR", "1e-6"))
    if not 0 <= margin <= 10 or not 0 < floor <= 1:
        raise ValueError("OCR 排序门槛无效")
    return {"margin": margin, "floor": floor}
