"""本地字符检测补充与 OCR 交叉核对。仅使用已安装的固定版本模型。"""
from pathlib import Path

import numpy as np
from PIL import Image


def package_model(name):
    import ddddocr
    return str(Path(ddddocr.__file__).parent/name)


def detect_characters(vision, image):
    from ddddocr.core.detection_engine import DetectionEngine

    class CPUDetector(DetectionEngine):
        def initialize(self, **kwargs):
            self.session = vision.session(package_model('common_det.onnx'))
            self.is_initialized = True

    if not hasattr(vision, '_text_detector'):
        vision._text_detector = CPUDetector()
    return vision._text_detector.predict(Image.fromarray(image[:, :, ::-1]))


def inspect_characters(vision, image, boxes):
    from ddddocr.models.charset_manager import CharsetManager
    if not hasattr(vision, '_ocr_charset'):
        manager = CharsetManager()
        manager.load_default_charset(beta=True)
        vision._ocr_charset = manager.charset
    chars = vision._ocr_charset
    session = vision.session(package_model('common.onnx'))
    results = []
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = Image.fromarray(image[y1:y2, x1:x2, ::-1])
        readings = []
        for angle in (-60, -30, 0, 30, 60):
            im = crop.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor='white').convert('L')
            im = im.resize((max(1, round(im.width*64/im.height)), 64), Image.Resampling.LANCZOS)
            tensor = np.asarray(im, dtype=np.float32)[None, None]/255
            raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
            if raw.ndim != 3 or raw.shape[1] != 1 or raw.shape[2] != len(chars):
                raise ValueError('OCR 模型输出与字符表不匹配')
            raw = raw[:, 0, :]
            exp = np.exp(raw-raw.max(axis=-1, keepdims=True))
            probability = exp/exp.sum(axis=-1, keepdims=True)
            peak = probability.max(axis=0)
            peak[0] = 0  # CTC 空白不是字符证据。
            index = int(peak.argmax())
            readings.append({'text': chars[index], 'score': float(peak[index]), 'angle': angle})
        results.append(choose_anchor(readings))
    return results


def choose_anchor(readings):
    """至少两个角度支持同一汉字，且没有高分冲突，才约束匹配。"""
    groups = {}
    for item in readings:
        char = item['text']
        if len(char) == 1 and '\u4e00' <= char <= '\u9fff':
            groups.setdefault(char, []).append(item['score'])
    ranked = sorted(((max(v), k, sorted(v, reverse=True)) for k, v in groups.items()), reverse=True)
    trusted = None
    confidence = 0.0
    if ranked:
        confidence, char, votes = ranked[0]
        alternative = ranked[1][0] if len(ranked)>1 else 0
        if len(votes)>=2 and confidence>=.95 and votes[1]>=.80 and confidence-alternative>=.15:
            trusted = char
    return {'trusted': trusted, 'score': confidence, 'readings': readings}


def constrain_scores(scores, prompt, observations):
    combined = np.array(scores, copy=True)
    anchored = {o['trusted'] for o in observations if o['trusted']}
    for i, char in enumerate(prompt):
        if char in anchored:
            combined[i, :] = 1e-6
    for j, observation in enumerate(observations):
        if observation['trusted']:
            for i, char in enumerate(prompt):
                combined[i, j] = observation['score'] if char == observation['trusted'] else 1e-6
    return combined
