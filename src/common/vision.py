"""本地 CPU 识别；输出候选值，不将模型分数解释为网站通过率。"""
from dataclasses import asdict, dataclass, field
from itertools import permutations
from pathlib import Path
import math
import os
import time

import cv2
import numpy as np


@dataclass
class Prediction:
    kind: str
    usable: bool
    score: float = 0.0
    points: list = field(default_factory=list)
    value: float | None = None
    reason: str = ""
    detail: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def to_dict(self):
        return plain_json(asdict(self))


def plain_json(value):
    if isinstance(value, np.ndarray):
        return plain_json(value.tolist())
    if isinstance(value, np.generic):
        return plain_json(value.item())
    if isinstance(value, dict):
        return {k: plain_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def best_assignment(scores):
    """提示数很少，精确枚举一对一匹配；同时保留第二名用于歧义检查。"""
    a = np.asarray(scores, dtype=float)
    if a.ndim != 2 or not 1 <= a.shape[0] <= 5 or not a.shape[0] <= a.shape[1] <= 8:
        raise ValueError("检测数量不支持：需要 1–5 个提示和不超过 8 个候选")
    if not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError("相似度矩阵无效")
    ranked = sorted(((float(sum(math.log(max(a[i, j], 1e-9)) for i, j in enumerate(p))), p)
                     for p in permutations(range(a.shape[1]), a.shape[0])), reverse=True)
    total, indices = ranked[0]
    gap = (total - ranked[1][0]) / a.shape[0] if len(ranked) > 1 else float("inf")
    return list(indices), float(min(a[i, j] for i, j in enumerate(indices))), gap


def letterbox(image, width, height, fill=114, interpolation=cv2.INTER_LINEAR):
    h, w = image.shape[:2]
    if h < 1 or w < 1:
        raise ValueError("空图像")
    ratio = min(width / w, height / h)
    nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
    x, y = (width - nw) // 2, (height - nh) // 2
    out = np.full((height, width, 3), fill, dtype=np.uint8)
    out[y:y + nh, x:x + nw] = cv2.resize(image, (nw, nh), interpolation=interpolation)
    tensor = np.ascontiguousarray(out[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    return tensor, ratio, (x, y)


class Vision:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.sessions = {}

    def session(self, name):
        if name not in self.sessions:
            import onnxruntime as ort
            ort.disable_telemetry_events()
            options = ort.SessionOptions()
            options.intra_op_num_threads = int(os.environ.get("CAPTCHA_CPU_THREADS", "2"))
            options.log_severity_level = 3
            self.sessions[name] = ort.InferenceSession(str(self.model_dir / name),
                                                       sess_options=options,
                                                       providers=["CPUExecutionProvider"])
        return self.sessions[name]

    def detections(self, image):
        from text.solver import detections
        return detections(self, image)

    def text(self, image, minimum=None, ambiguity=None, hint_boxes=None, target_region=None, hint_text=None, detector="default", ocr_check=True):
        from text.solver import recognize
        return recognize(self, image, minimum, ambiguity, hint_boxes, target_region, hint_text, detector, ocr_check)

    def rotate_single(self, image):
        from rotate_single.solver import predict_angle
        return predict_angle(self, image)

    def slider(self, background: bytes, piece: bytes):
        from slider.solver import baseline
        return baseline(background, piece)

    def rotate_dual(self, background, piece):
        from rotate_dual.solver import predict
        return predict(background, piece)


def _finish(kind, start, usable=False, score=0., **kwargs):
    return Prediction(kind, bool(usable), float(score), elapsed_ms=(time.monotonic()-start)*1000, **kwargs)
