"""单图绝对朝向识别与旋转一致性检查。"""
import math
import time
import cv2
import numpy as np
from common.vision import Prediction


def predict_angle(vision, image):
    start = time.monotonic()
    h, w = image.shape[:2]
    side = int(min(h, w) / math.sqrt(2))
    x, y = (w - side) // 2, (h - side) // 2
    a = cv2.resize(image[y:y + side, x:x + side], (224, 224))[:, :, ::-1].astype(np.float32) / 255
    a = (a - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    session = vision.session("rotate.onnx")
    raw = session.run(None, {session.get_inputs()[0].name: np.ascontiguousarray(a.transpose(2, 0, 1)[None])})[0].ravel()
    probs = np.exp(raw - raw.max())
    probs /= probs.sum()
    index = int(np.argmax(probs))
    # 分类角度方向和网页拖动映射必须在对应站点标定后才能自动操作。
    return Prediction("rotate_single", False, float(probs[index]), value=index * 360 / len(raw),
                      reason="离线角度已输出；线上拖动映射和置信度尚未标定",
                      detail={"classes": len(raw), "top3": np.argsort(probs)[-3:][::-1].tolist(),
                              "mass_within_8_degrees": float(probs[np.abs((np.arange(len(raw))*360/len(raw)-index*360/len(raw)+180)%360-180)<=8].sum())},
                      elapsed_ms=(time.monotonic() - start) * 1000)


def upright(vision, image):
    start = time.monotonic()
    a = vision.rotate_single(image)
    b = vision.rotate_single(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    delta = abs((b.value - 90 - a.value + 180) % 360 - 180)
    mass = min(a.detail['mass_within_8_degrees'], b.detail['mass_within_8_degrees'])
    usable = mass >= .65 and delta <= 8
    return Prediction('rotate_single', usable, mass, value=a.value,
                      reason='角度集中且旋转一致' if usable else '朝向分布分散或旋转后预测不一致',
                      detail={'clockwise_correction': a.value, 'quarter_turn_error': delta,
                              'mass_floor': .65, 'consistency_limit_degrees': 8,
                              'views': [a.to_dict(), b.to_dict()],
                              'scope': '绝对朝向；分数不是校准正确率，网页映射由适配器保证'},
                      elapsed_ms=(time.monotonic()-start)*1000)
