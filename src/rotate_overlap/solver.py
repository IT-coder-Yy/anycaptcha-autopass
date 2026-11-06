"""完整底图与中心裁片的重叠旋转配准。"""
import time
import cv2
import numpy as np
from common.vision import Prediction


def align_overlap(background, piece):
    """对应 GoCaptcha 原图与中心裁片，比较共同半径上的角向纹理。

    此模式要求两图包含相同像素区域，与黑色缺口双圈、单图摆正不同。
    返回让内盘顺时针转动的角度。分数是相关系数，不是通过概率。
    """
    start = time.monotonic()
    if min(piece.shape[:2]) < 32 or any(a < b for a, b in zip(background.shape[:2], piece.shape[:2])):
        return Prediction("rotate_overlap", False, reason="原图与裁片尺寸不符合重叠旋转条件")
    size = 720
    theta = np.arange(size, dtype=np.float32) * 2 * np.pi / size
    radius = np.linspace(8, min(piece.shape[:2]) * .43, 32, dtype=np.float32)

    def sample(im):
        h, w = im.shape[:2]
        x = (w - 1) / 2 + radius[:, None] * np.cos(theta)[None, :]
        y = (h - 1) / 2 + radius[:, None] * np.sin(theta)[None, :]
        a = cv2.remap(im[:, :, :3].astype(np.float32), x, y, cv2.INTER_LINEAR)
        a -= a.mean(axis=1, keepdims=True)
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-5)

    a, b = sample(background), sample(piece)
    curve = np.fft.irfft(np.fft.rfft(a, axis=1) * np.conj(np.fft.rfft(b, axis=1)),
                        n=size, axis=1).mean(axis=(0, 2))
    index = int(curve.argmax())
    second = curve.copy()
    for delta in range(-20, 21):
        second[(index + delta) % size] = -1
    score, margin = float(curve[index]), float(curve[index] - second.max())
    usable = score >= .80 and margin >= .05
    return Prediction("rotate_overlap", usable, score, value=index * 360 / size,
                      reason="重叠图像匹配" if usable else "图像不匹配或旋转角度有歧义",
                      detail={"margin": margin, "mode": "overlapping_center_crop"},
                      elapsed_ms=(time.monotonic() - start) * 1000)
