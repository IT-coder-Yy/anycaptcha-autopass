"""滑块图像候选：轮廓匹配补充旧内容/填充投票，分数不是通过概率。"""
from dataclasses import asdict
import math
import os
import time

import cv2
import numpy as np

from slider.detect import detect_gap
from slider.geetest import locate_slice
from common.vision import Prediction


def settings():
    engine = os.environ.get('CAPTCHA_SLIDER_ENGINE', 'outline')
    minimum = float(os.environ.get('CAPTCHA_SLIDER_OUTLINE_MIN', '.18'))
    margin = float(os.environ.get('CAPTCHA_SLIDER_OUTLINE_MARGIN', '.06'))
    if engine not in {'baseline', 'outline'} or not 0 < minimum < 1 or not 0 < margin < 1:
        raise ValueError('滑块方案或轮廓阈值无效')
    return {'engine': engine, 'minimum': minimum, 'margin': margin}


def outline_candidate(background, piece, cur_x=0., cur_y=0., *, minimum=.18, margin=.06):
    """用完整 alpha 轮廓匹配边缘；保留透明边框，限制水平行带及可达行程。"""
    if piece.ndim != 3 or piece.shape[2] != 4:
        return Prediction('slider', False, reason='轮廓匹配需要透明通道')
    mask = (piece[:, :, 3] >= 128).astype(np.uint8)*255
    ys, xs = np.nonzero(mask)
    if len(xs) < 60:
        return Prediction('slider', False, reason='可见碎片像素不足')
    left, top = max(0, int(xs.min())-3), max(0, int(ys.min())-3)
    right = min(mask.shape[1], int(xs.max())+4)
    bottom = min(mask.shape[0], int(ys.max())+4)
    template = cv2.Canny(mask[top:bottom, left:right], 50, 150)
    if np.count_nonzero(template) < 24:
        return Prediction('slider', False, reason='轮廓信息不足')
    gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    expected_y = round(cur_y+top)
    low = max(0, expected_y-3)
    high = min(background.shape[0], expected_y+template.shape[0]+3)
    first_x = max(0, math.ceil(cur_x+left+1))
    last_x = math.floor(background.shape[1]-piece.shape[1]+left)
    if high-low < template.shape[0] or last_x <= first_x:
        return Prediction('slider', False, reason='没有可达的水平搜索区域')
    response = cv2.matchTemplate(edges[low:high], template, cv2.TM_CCOEFF_NORMED)
    response = np.nan_to_num(response, nan=-1., posinf=-1., neginf=-1.)
    response[:, :first_x] = -1
    response[:, last_x+1:] = -1
    _, peak, _, (x, y) = cv2.minMaxLoc(response)
    rivals = response.copy()
    guard = max(8, template.shape[1]//2)
    rivals[:, max(0, x-guard):x+guard+1] = -1
    second = max(0., float(rivals.max()))
    usable = peak >= minimum and peak-second >= margin
    return Prediction('slider', usable, float(peak), value=float(x-left-cur_x),
                      reason='完整轮廓峰值与次峰差满足门槛' if usable else '轮廓弱或存在近似候选',
                      detail={'peak': float(peak), 'second': second, 'margin': float(peak-second),
                              'y_offset': float(low+y-(cur_y+top)),
                              'template_box': [left, top, right-left, bottom-top],
                              'score_semantics': 'outline_edge_correlation'})


def predict_slider(background, piece, site, cur_x=0., cur_y=0., *, engine=None):
    started = time.monotonic()
    config = settings()
    engine = config['engine'] if engine is None else engine
    if engine not in {'baseline', 'outline'}:
        raise ValueError('未知滑块方案')
    if site == 'yidun':
        result = locate_slice(background, piece, cur_x, cur_y)
        baseline = Prediction('slider', result.move is not None and result.conf >= .45,
                              result.conf, value=result.move, reason=result.note,
                              detail={'candidates': [asdict(c) for c in result.candidates]})
    elif site == 'kg':
        result = detect_gap(cv2.imencode('.png', background)[1].tobytes(),
                            cv2.imencode('.png', piece)[1].tobytes(), None, 12, False)
        move = None if result.x is None else result.x-result.piece_box[0]-cur_x
        baseline = Prediction('slider', move is not None and result.conf >= .5, result.conf, value=move,
                              detail={'piece_box': result.piece_box,
                                      'candidates': [asdict(c) for c in result.candidates]})
    else:
        raise ValueError('滑块尚未适配该站点')
    selected = baseline
    detail = {'engine': engine, 'baseline': baseline.to_dict(), 'decision': 'baseline'}
    if engine == 'outline':
        outline = outline_candidate(background, piece, cur_x, cur_y,
                                    minimum=config['minimum'], margin=config['margin'])
        detail['outline'] = outline.to_dict()
        if outline.usable:
            if site == 'yidun' and baseline.usable:
                if abs(outline.value-baseline.value) > 4:
                    selected = Prediction('slider', False, reason='内容与轮廓位置分歧超过 4px')
                    detail['decision'] = 'conflict'
                else:
                    detail['decision'] = 'content_confirmed_by_outline'
            else:
                selected = outline
                detail['decision'] = 'outline'
    return Prediction('slider', selected.usable, selected.score, value=selected.value,
                      reason=selected.reason, detail=detail,
                      elapsed_ms=(time.monotonic()-started)*1000)


def baseline(background: bytes, piece: bytes):
    from slider.detect import detect_gap
    start = time.monotonic()
    result = detect_gap(background, piece, want_debug=False)
    return Prediction("slider", result.x is not None and result.conf >= .5, result.conf,
                      value=result.x, reason="缺口候选，页面拖动映射需单独校准",
                      detail={"y": result.y, "piece_box": result.piece_box,
                              "candidates": [asdict(c) for c in result.candidates]},
                      elapsed_ms=(time.monotonic() - start) * 1000)
