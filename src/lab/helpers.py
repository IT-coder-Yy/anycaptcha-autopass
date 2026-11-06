"""本地题库共用的图像编码和背景生成。"""
import base64
import cv2
import numpy as np


def _png(image):
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('无法编码题图')
    return 'data:image/png;base64,'+base64.b64encode(data).decode()

def _texture(rng, width, height):
    # 多尺度色块与独立曲线；生成器不使用识别器的匹配函数。
    seed = rng.getrandbits(63)
    nr = np.random.default_rng(seed)
    coarse = nr.integers(25,230,(12,18,3),dtype=np.uint8)
    image = cv2.resize(coarse,(width,height),interpolation=cv2.INTER_CUBIC)
    for _ in range(12):
        points = np.array([[rng.randrange(width),rng.randrange(height)] for _ in range(4)],np.int32)
        cv2.polylines(image,[points],False,tuple(rng.randrange(30,230) for _ in range(3)),rng.randrange(1,4),cv2.LINE_AA)
    return image
