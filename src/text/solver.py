"""顺序点字的字符检测与匹配。"""
import math
import os
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
from common.vision import Prediction, letterbox, best_assignment


def detections(vision, image):
    session = vision.session("text-detector.onnx")
    inp = session.get_inputs()[0]
    tensor, scale, (ox, oy) = letterbox(image, inp.shape[3], inp.shape[2])
    raw = session.run(None, {inp.name: tensor})[0]
    if raw.ndim != 3 or raw.shape[-1] != 6:
        raise ValueError("检测模型输出格式变化")
    found = []
    for x1, y1, x2, y2, confidence, label in raw[0]:
        if confidence < .3:
            continue
        box = [max(0, int((x1 - ox) / scale)), max(0, int((y1 - oy) / scale)),
               min(image.shape[1], int((x2 - ox) / scale)), min(image.shape[0], int((y2 - oy) / scale))]
        if box[2] > box[0] and box[3] > box[1]:
            found.append({"box": box, "score": float(confidence), "class": int(label)})
    return found


def recognize(vision, image, minimum=None, ambiguity=None, hint_boxes=None, target_region=None, hint_text=None, detector="default", ocr_check=True):
    minimum=float(os.environ.get("CAPTCHA_TEXT_THRESHOLD","0.80")) if minimum is None else minimum
    ambiguity=float(os.environ.get("CAPTCHA_TEXT_AMBIGUITY","0.08")) if ambiguity is None else ambiguity
    if not 0<=minimum<=1 or not 0<=ambiguity<=10:
        raise ValueError("点字阈值或歧义阈值无效")
    start = time.monotonic()
    found = vision.detections(image)
    targets = [v["box"] for v in found if v["class"] == 0]
    if detector == "ddddocr" and target_region:
        from common.text_support import detect_characters
        rx,ry,rw,rh=map(round,target_region)
        targets=[[a+rx,b+ry,c+rx,d+ry] for a,b,c,d in detect_characters(vision,image[ry:ry+rh,rx:rx+rw])]
    if target_region:
        rx, ry, rw, rh = target_region
        targets = [b for b in targets if rx <= (b[0]+b[2])/2 < rx+rw and ry <= (b[1]+b[3])/2 < ry+rh]
    hints = hint_boxes if hint_boxes is not None else sorted((v["box"] for v in found if v["class"] == 2), key=lambda b: b[0])
    hints = [[max(0, int(b[0])), max(0, int(b[1])), min(image.shape[1], int(math.ceil(b[2]))),
              min(image.shape[0], int(math.ceil(b[3])))] for b in hints]
    detail = {"detections": found, "hint_boxes": hints, "target_boxes": targets,
              "prompt_count": len(hints), "target_count": len(targets), "hint_text":hint_text,"detector":detector}
    if not 1 <= len(hints) <= 5 or not len(hints) <= len(targets) <= 8:
        return Prediction("text", False, reason="提示或候选字符检测不完整", detail=detail,
                          elapsed_ms=(time.monotonic() - start) * 1000)
    def crop(box):
        x1, y1, x2, y2 = box
        return letterbox(image[y1:y2, x1:x2], 112, 112, 128, cv2.INTER_CUBIC)[0]
    left = np.concatenate([crop(b) for b in hints])
    if hint_text and len(hint_text)==len(hints):
        from PIL import Image, ImageDraw, ImageFont
        font_path=os.environ.get("CAPTCHA_FONT")
        if not font_path:
            try:
                font_path=subprocess.check_output(["fc-match",":lang=zh","-f","%{file}"],text=True,timeout=3).strip()
            except (OSError,subprocess.SubprocessError):
                font_path=None
        if font_path and Path(font_path).is_file():
            font=ImageFont.truetype(font_path,64)
            clean=[]
            for char in hint_text:
                glyph=Image.new("RGB",(90,90),"white")
                drawing=ImageDraw.Draw(glyph)
                x0,y0,x1,y1=drawing.textbbox((0,0),char,font=font)
                drawing.text(((90-x1+x0)/2-x0,(90-y1+y0)/2-y0),char,fill="black",font=font)
                clean.append(letterbox(np.array(glyph)[:,:,::-1],112,112,128,cv2.INTER_CUBIC)[0])
            left=np.concatenate(clean)
            detail["hint_source"]="visible_dom_text_rendered_locally"
    right = np.concatenate([crop(b) for b in targets])
    session = vision.session("text-matcher.onnx")
    inputs = session.get_inputs()
    raw = session.run(None, {inputs[0].name: np.repeat(left, len(targets), axis=0),
                             inputs[1].name: np.tile(right, (len(hints), 1, 1, 1))})[0]
    scores = (1 / (1 + np.exp(-np.clip(raw, -50, 50)))).reshape(len(hints), len(targets))
    if ocr_check and hint_text and len(hint_text)==len(hints):
        from common.text_support import inspect_characters, constrain_scores
        observations=inspect_characters(vision,image,targets)
        detail['raw_similarities']=scores.tolist()
        detail['ocr']=observations
        scores=constrain_scores(scores,hint_text,observations)
    indices, confidence, gap = best_assignment(scores)
    detail.update({"similarities": scores.tolist(), "assignment": indices,
                   "assignment_margin": gap if math.isfinite(gap) else None})
    usable = confidence >= minimum and gap >= ambiguity
    engine = os.environ.get("CAPTCHA_TEXT_ENGINE", "consensus")
    if engine not in {"siamese", "consensus"}:
        raise ValueError("未知点字识别器")
    detail["engine"] = engine
    if engine == "consensus" and hint_text and len(hint_text) == len(hints):
        from common.text_ocr import rank_characters, choose_assignment, settings
        models = rank_characters(vision, image, targets, hint_text)
        detail["baseline"] = {"assignment": indices, "score": confidence, "margin": gap, "usable": usable}
        indices, usable, method = choose_assignment(models, indices, usable, **settings())
        if method == "ppocr_consensus":
            confidence = min(m["score"] for m in models)
            gap = min(m["margin"] for m in models)
        detail.update({"ocr_models": models, "decision": method, "assignment": indices,
                       "assignment_margin": gap if math.isfinite(gap) else None,
                       "score_semantics": "minimum_selected_ocr_probability" if method == "ppocr_consensus"
                                          else "baseline_similarity"})
    points = [[(targets[j][0] + targets[j][2]) / 2, (targets[j][1] + targets[j][3]) / 2] for j in indices]
    return Prediction("text", usable, confidence, points=points,
                      reason="可评测候选" if usable else "无法得到足够支持的字符排列", detail=detail,
                      elapsed_ms=(time.monotonic() - start) * 1000)
