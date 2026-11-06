"""单碎片二维配准：返回碎片原图坐标。"""
import time
import cv2
import numpy as np
from common.vision import _finish


def _masked_correlation(background, template, mask):
    """掩码内零均值相关，对缺口的统一压暗/提亮不敏感。"""
    mask = mask.astype(np.float32)
    n = float(mask.sum())
    numerator = variance = None
    energy = 0.
    for channel in range(3):
        b = background[:, :, channel].astype(np.float32)
        t = template[:, :, channel].astype(np.float32)
        centered = (t-float((t*mask).sum()/n))*mask
        term = cv2.matchTemplate(b, centered, cv2.TM_CCORR)
        sums = cv2.matchTemplate(b, mask, cv2.TM_CCORR)
        squares = cv2.matchTemplate(b*b, mask, cv2.TM_CCORR)
        var = np.maximum(squares-sums*sums/n, 0)
        numerator = term if numerator is None else numerator+term
        variance = var if variance is None else variance+var
        energy += float((centered*centered).sum())
    if energy < n:
        return None
    return np.clip(numerator/np.sqrt(np.maximum(variance*energy, 1e-8)), -1, 1)

def drag_piece(background, piece):
    """返回整个透明碎片图左上角在原背景中的目标坐标，非鼠标坐标。"""
    start = time.monotonic()
    if piece.ndim != 3 or piece.shape[2] != 4:
        return _finish('drag2d', start, reason='二维拼图需要含透明通道的碎片')
    mask = (piece[:, :, 3] > 200).astype(np.uint8)
    if not mask.any():
        return _finish('drag2d', start, reason='碎片为空')
    x, y, w, h = cv2.boundingRect(mask)
    if piece.shape[1] > background.shape[1] or piece.shape[0] > background.shape[0] or min(w, h) < 8:
        return _finish('drag2d', start, reason='碎片与背景尺寸不兼容')
    inner = cv2.erode(mask[y:y+h, x:x+w], np.ones((5,5), np.uint8))
    if inner.sum() < 30:
        return _finish('drag2d', start, reason='碎片有效纹理面积不足')
    # 保留完整透明边界，先对齐缺口轮廓，再用纹理作辅助。
    outline = (cv2.Canny(mask*255,50,100)>0).astype(np.float32)
    edges = cv2.Canny(background[:, :, :3],80,160)
    distances = cv2.distanceTransform(255-edges,cv2.DIST_L2,3)
    edge_cost = cv2.matchTemplate(distances,outline,cv2.TM_CCORR)/max(1,float(outline.sum()))
    texture = _masked_correlation(background[:, :, :3], piece[:, :, :3],
                                  cv2.erode(mask,np.ones((5,5),np.uint8)))
    contour = np.exp(-edge_cost)
    curve = contour if texture is None else .85*contour+.15*np.maximum(texture,0)
    cy, cx = np.unravel_index(int(curve.argmax()), curve.shape)
    score = float(curve[cy,cx])
    other = curve.copy()
    radius = max(5, min(w,h)//4)
    other[max(0,cy-radius):cy+radius+1,max(0,cx-radius):cx+radius+1] = -1
    margin = score-float(other.max())
    usable = score >= .55 and margin >= .04
    return _finish('drag2d', start, usable, score, points=[[float(cx),float(cy)]],
                   reason='二维纹理配准候选' if usable else '二维候选相关性不足或存在重复纹理',
                   detail={'margin':margin,'edge_cost':float(edge_cost[cy,cx]),
                           'texture_score':None if texture is None else float(texture[cy,cx]),
                           'piece_box':[x,y,w,h], 'coordinate':'piece_image_top_left'})
