"""易盾三图标：细化轮廓位置，结合可见像素对比度排除照片伪候选。"""
import itertools
import math
import time

import cv2
import numpy as np

from common.vision import Prediction
from icon.yidun_v1 import MIN_FIT, MIN_MARGIN, hint_masks, _background_masks, _variant, _area_sums
from icon.yidun_v1 import click_yidun_icons as click_v1

ENGINE = 'yidun-contour-v2'


def _photometric(image, mask, candidate):
    kernel = _variant(mask, candidate['angle'], candidate['scale'])
    h, w = kernel.shape
    x, y = candidate['point']
    patch = image[y-h//2:y-h//2+h, x-w//2:x-w//2+w, :3]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(float)
    centered = kernel-kernel.mean()
    values = gray-gray.mean()
    correlation = abs(np.sum(centered*values))/max(1, np.linalg.norm(centered)*np.linalg.norm(values))
    core = cv2.erode(kernel, np.ones((3, 3), np.uint8)) > .5
    if core.sum() < 10:
        core = kernel > .5
    samples = patch[core].astype(float)
    spread = float(np.quantile(np.linalg.norm(samples-np.median(samples, axis=0), axis=1), .75))
    # 平坦天空也可能低方差，故颜色一致性仅作轻量惩罚，不能替代轮廓和前景/背景对比。
    candidate.update(correlation=float(correlation), spread=spread,
                     evidence=float(np.sqrt(candidate['score']*correlation/(1+spread/30)**.25)))
    return candidate


def _refine(image, mask, candidate, backgrounds, blocked):
    x, y = candidate['point']
    best = dict(candidate)
    for angle in range(candidate['angle']-8, candidate['angle']+9, 2):
        for scale in np.arange(candidate['scale']-.15, candidate['scale']+.151, .05):
            kernel = _variant(mask, angle, scale)
            h, w = kernel.shape
            left, top = max(0, x-w//2-3), max(0, y-h//2-3)
            right, bottom = min(image.shape[1], x-w//2+w+3), min(image.shape[0], y-h//2+h+3)
            if bottom-top < h or right-left < w:
                continue
            allowed = _area_sums(cv2.integral(blocked[top:bottom, left:right]), h, w) == 0
            for background in backgrounds:
                patch = background[top:bottom, left:right]
                intersection = cv2.matchTemplate(patch, kernel, cv2.TM_CCORR)
                area = _area_sums(cv2.integral(patch), h, w)
                scores = intersection/np.maximum(1, kernel.sum()+area-intersection)
                scores[~allowed] = 0
                yy, xx = np.unravel_index(scores.argmax(), scores.shape)
                if scores[yy, xx] > best['score']:
                    best.update(point=[left+int(xx)+w//2, top+int(yy)+h//2],
                                score=float(scores[yy, xx]), angle=angle, scale=float(scale))
    return _photometric(image, mask, best)


def _candidates(image, mask, backgrounds, integrals, blocked):
    best = np.zeros(image.shape[:2], np.float32)
    angles = np.zeros(best.shape, np.int16)
    scales = np.zeros(best.shape, np.float32)
    blocked_integral = cv2.integral(blocked)
    for angle in range(0, 360, 10):
        for scale in np.arange(1.25, 2.26, .2):
            kernel = _variant(mask, angle, scale)
            h, w = kernel.shape
            if h > best.shape[0] or w > best.shape[1]:
                continue
            region = np.s_[h//2:h//2+best.shape[0]-h+1, w//2:w//2+best.shape[1]-w+1]
            destination, aa, ss = best[region], angles[region], scales[region]
            allowed = _area_sums(blocked_integral, h, w) == 0
            for background, integral in zip(backgrounds, integrals):
                intersection = cv2.matchTemplate(background, kernel, cv2.TM_CCORR)
                area = _area_sums(integral, h, w)
                scores = intersection/np.maximum(1, kernel.sum()+area-intersection)
                scores[~allowed] = 0
                improved = scores > destination
                aa[improved], ss[improved] = angle, scale
                destination[improved] = scores[improved]
    candidates = []
    for _ in range(8):
        y, x = np.unravel_index(best.argmax(), best.shape)
        if best[y, x] <= 0:
            break
        candidate = {'point':[int(x), int(y)], 'score':float(best[y, x]),
                     'angle':int(angles[y, x]), 'scale':float(scales[y, x])}
        candidates.append(_refine(image, mask, candidate, backgrounds, blocked))
        best[max(0, y-20):y+21, max(0, x-20):x+21] = 0
    # 细化后同一位置可能合并，保留证据较强的一项；不同物体仍参与次佳排列。
    distinct = []
    for candidate in sorted(candidates, key=lambda c:c['evidence'], reverse=True):
        if all(math.dist(candidate['point'], other['point']) >= 20 for other in distinct):
            distinct.append(candidate)
    return distinct


def click_yidun_icons(background, hint, ignored_rects=()):
    start = time.monotonic()
    initial = click_v1(background, hint, ignored_rects)
    # 保留已验收的第一阶段判据，只对弃权题增加证据，不重排其已接受结果。
    if initial.usable:
        initial.detail.update(engine=ENGINE, stage='v1-accepted')
        return initial
    detail = {'engine':ENGINE, 'min_fit':MIN_FIT, 'min_margin':MIN_MARGIN, 'ignored_rects':ignored_rects,
              'stage':'refined', 'initial_score':initial.score, 'initial_margin':initial.detail.get('margin'),
              'ranking':'sqrt(iou * abs_grayscale_correlation / (1 + foreground_spread / 30) ** 0.25)'}

    def finish(usable=False, score=0., points=None, reason='提示分割不完整'):
        return Prediction('icon', bool(usable), float(score), points=points or [], reason=reason,
                          detail=detail, elapsed_ms=(time.monotonic()-start)*1000)

    if (background is None or hint is None or background.ndim != 3 or hint.ndim != 3
            or background.shape[2] not in (3, 4) or hint.shape[2] not in (3, 4)
            or min(background.shape[:2]) < 64 or max(background.shape[:2]) > 1000
            or not 12 <= hint.shape[0] <= 64 or not 40 <= hint.shape[1] <= 300):
        return finish(reason='需要可见背景和三图标提示截图')
    hints = hint_masks(hint)
    detail['hint_boxes'] = [box for box, _ in hints]
    if len(hints) != 3 or any(min(box[2:]) < 6 or max(box[2:]) > 32 for box, _ in hints):
        return finish()
    backgrounds, blocked = _background_masks(background, ignored_rects)
    integrals = [cv2.integral(mask) for mask in backgrounds]
    candidates = [_candidates(background, mask, backgrounds, integrals, blocked) for _, mask in hints]
    detail['candidates'] = candidates
    ranked = []
    for choice in itertools.product(*candidates):
        if any(math.dist(a['point'], b['point']) < 22 for index, a in enumerate(choice) for b in choice[:index]):
            continue
        ranked.append((sum(math.log(max(c['evidence'], 1e-6)) for c in choice), choice))
    ranked.sort(key=lambda item:item[0], reverse=True)
    if not ranked:
        return finish(reason='没有三个相互分离的目标候选')
    total, chosen = ranked[0]
    margin = float(total-ranked[1][0]) if len(ranked) > 1 else 1.
    scores = [item['score'] for item in chosen]
    detail.update(scores=scores, margin=margin, evidence=[item['evidence'] for item in chosen])
    usable = min(scores) >= MIN_FIT and margin >= MIN_MARGIN
    return finish(usable, min(scores), [item['point'] for item in chosen],
                  '轮廓与像素证据一致的排列候选' if usable else '轮廓匹配不足或位置排列歧义')
