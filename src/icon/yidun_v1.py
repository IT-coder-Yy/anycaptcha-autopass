"""易盾演示的三图标顺序匹配：从可见提示生成临时模板，不需要模型或物品字典。"""
import itertools
import time

import cv2
import numpy as np

from common.vision import Prediction


ENGINE = 'yidun-contour-v1'
MIN_FIT = .60
MIN_MARGIN = .12


def hint_masks(image):
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    mask = (gray < 150).astype(np.uint8)
    occupied = np.flatnonzero(mask.sum(axis=0) > 0)
    groups = np.split(occupied, np.where(np.diff(occupied) > 4)[0]+1)
    result = []
    for group in groups:
        if not len(group):
            continue
        left, right = int(group[0]), int(group[-1])+1
        part = mask[:, left:right]
        if part.sum() < 15:
            continue
        x, y, w, h = cv2.boundingRect(part)
        # 不再删除贴边图标，不做会吞掉细线/小孔的开闭运算。
        result.append(([left+x, y, w, h], part[y:y+h, x:x+w]))
    return result


def _background_masks(image, ignored_rects):
    rgb = image[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    chroma = rgb.max(axis=2).astype(int)-rgb.min(axis=2)
    masks = [((gray < limit) & (chroma < 50)).astype(np.float32) for limit in (50, 70, 90, 110)]
    masks += [((gray > limit) & (chroma < 70)).astype(np.float32) for limit in (155, 180, 205, 225)]
    blocked = np.zeros(gray.shape, np.uint8)
    for x, y, w, h in ignored_rects:
        left, top = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
        right, bottom = min(gray.shape[1], int(np.ceil(x+w))), min(gray.shape[0], int(np.ceil(y+h)))
        if right > left and bottom > top:
            blocked[top:bottom, left:right] = 1
    for mask in masks:
        mask[blocked > 0] = 0
    return masks, blocked


def _variant(mask, angle, scale):
    # 画布足够容纳最大尺度及对角旋转，避免截断模板。
    padded = cv2.copyMakeBorder(mask, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=0)
    h, w = padded.shape
    rotated = cv2.warpAffine(padded.astype(np.float32),
                            cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, scale), (w, h))
    rotated = (rotated > .45).astype(np.float32)
    x, y, w, h = cv2.boundingRect(rotated.astype(np.uint8))
    return cv2.copyMakeBorder(rotated[y:y+h, x:x+w], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)


def _area_sums(integral, h, w):
    return integral[h:, w:]-integral[:-h, w:]-integral[h:, :-w]+integral[:-h, :-w]


def _candidates(mask, backgrounds, integrals, blocked):
    best = np.zeros(backgrounds[0].shape, np.float32)
    blocked_integral = cv2.integral(blocked)
    for angle in range(0, 360, 10):
        for scale in np.arange(1.25, 2.26, .2):
            kernel = _variant(mask, angle, scale)
            h, w = kernel.shape
            if h > best.shape[0] or w > best.shape[1]:
                continue
            total = kernel.sum()
            allowed = _area_sums(blocked_integral, h, w) == 0
            for background, integral in zip(backgrounds, integrals):
                intersection = cv2.matchTemplate(background, kernel, cv2.TM_CCORR)
                area = _area_sums(integral, h, w)
                # 四周空白也参与并集，防止只匹配到大色块的局部而冒充图标。
                score = intersection/np.maximum(1, total+area-intersection)
                score[~allowed] = 0
                destination = best[h//2:h//2+score.shape[0], w//2:w//2+score.shape[1]]
                np.maximum(destination, score, out=destination)
    result = []
    for _ in range(5):
        y, x = np.unravel_index(best.argmax(), best.shape)
        score = float(best[y, x])
        if score <= 0:
            break
        result.append({'point':[int(x), int(y)], 'score':score})
        best[max(0, y-20):y+21, max(0, x-20):x+21] = 0
    return result


def click_yidun_icons(background, hint, ignored_rects=()):
    start = time.monotonic()
    detail = {'engine':ENGINE, 'min_fit':MIN_FIT, 'min_margin':MIN_MARGIN, 'ignored_rects':ignored_rects}

    def finish(usable=False, score=0., points=None, reason='提示分割不完整'):
        return Prediction('icon', bool(usable), float(score), points=points or [], reason=reason,
                          detail=detail, elapsed_ms=(time.monotonic()-start)*1000)

    if (background is None or hint is None or background.ndim != 3 or hint.ndim != 3
            or background.shape[2] not in (3, 4) or hint.shape[2] not in (3, 4)
            or min(background.shape[:2]) < 64 or max(background.shape[:2]) > 1000
            or not 12 <= hint.shape[0] <= 64 or not 40 <= hint.shape[1] <= 300):
        return finish(reason='需要可见背景和三图标提示截图')
    hints = hint_masks(hint)
    detail['hint_boxes'] = [box for box, mask in hints]
    if len(hints) != 3 or any(min(box[2:]) < 6 or max(box[2:]) > 32 for box, mask in hints):
        return finish()
    backgrounds, blocked = _background_masks(background, ignored_rects)
    integrals = [cv2.integral(mask) for mask in backgrounds]
    candidates = [_candidates(mask, backgrounds, integrals, blocked) for box, mask in hints]
    detail['candidates'] = candidates
    ranked = []
    for choice in itertools.product(*candidates):
        if any(np.linalg.norm(np.array(a['point'])-b['point']) < 22
               for index, a in enumerate(choice) for b in choice[:index]):
            continue
        ranked.append((sum(np.log(max(item['score'], 1e-6)) for item in choice), choice))
    ranked.sort(key=lambda item:item[0], reverse=True)
    if not ranked:
        return finish(reason='没有三个相互分离的目标候选')
    total, chosen = ranked[0]
    margin = float(total-ranked[1][0]) if len(ranked) > 1 else 1.
    scores = [item['score'] for item in chosen]
    detail.update(scores=scores, margin=margin)
    usable = min(scores) >= MIN_FIT and margin >= MIN_MARGIN
    return finish(usable, min(scores), [item['point'] for item in chosen],
                  '图标轮廓排列候选' if usable else '轮廓匹配不足或位置排列歧义')
