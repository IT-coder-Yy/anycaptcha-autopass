"""横带滑动还原：接缝匹配与平移候选。"""
import time
import numpy as np
from common.vision import _finish


def restore_band(image, band):
    """横带环形平移；band=(x,y,w,h) 是页面可见的移动区域，不包含正确位移。"""
    start = time.monotonic()
    x, y, w, h = map(int, band)
    ih, iw = image.shape[:2]
    if w < 32 or h < 4 or x < 0 or y < 0 or x+w > iw or y+h > ih:
        return _finish('restore', start, reason='移动横带尺寸无效')
    a = image[:, :, :3].astype(np.float32)
    pairs = []
    if y > 0:
        pairs.append((a[y, x:x+w], a[y-1, x:x+w]))
    if y+h < ih:
        pairs.append((a[y+h-1, x:x+w], a[y+h, x:x+w]))
    if not pairs:
        return _finish('restore', start, reason='没有可比较的静止接缝')
    # 比较相邻行的原始颜色及沿 x 方向的梯度，不使用出题器的位移范围。
    cost = np.zeros(w)
    texture = float(np.mean([b.std(axis=0).mean() for _, b in pairs]))
    for moving, fixed in pairs:
        dm, df = np.gradient(moving, axis=0), np.gradient(fixed, axis=0)
        for shift in range(w):
            cost[shift] += .7*np.abs(np.roll(moving, shift, axis=0)-fixed).mean()
            cost[shift] += .3*np.abs(np.roll(dm, shift, axis=0)-df).mean()
    index = int(cost.argmin())
    other = cost.copy()
    other[(index + np.arange(-6, 7)) % w] = np.inf
    margin = float((other.min()-cost[index])/max(other.min(), 1e-6))
    score = float(1-cost[index]/max(float(np.median(cost)), 1e-6))
    usable = texture >= 4 and score >= .35 and margin >= .08
    return _finish('restore', start, usable, score, value=float(index),
                   reason='横带接缝候选' if usable else '接缝纹理不足或平移候选有歧义',
                   detail={'band':list(band), 'margin':margin, 'texture':texture})
