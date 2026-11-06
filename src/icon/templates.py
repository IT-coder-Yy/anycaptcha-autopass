"""GoCaptcha 公开 13 类形状的本地匹配；固定素材不是当前题目的答案。"""
from functools import lru_cache
import hashlib
from itertools import permutations
import json
from pathlib import Path
import time

import cv2
import numpy as np

from icon.solver import _hint_count
from common.vision import Prediction


ASSETS = Path(__file__).with_name('assets') / 'gocaptcha-shapes'
ENGINE = 'gocaptcha-template-v1'


@lru_cache(maxsize=1)
def template_info():
    manifest = json.loads((ASSETS / 'manifest.json').read_text())
    digest = hashlib.sha256()
    for item in manifest['files']:
        data = (ASSETS / item['name']).read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('图标模板校验失败：' + item['name'])
        digest.update(data)
    return {'engine': ENGINE, 'repository': manifest['repository'],
            'revision': manifest['revision'], 'templates_sha256': digest.hexdigest()}


def _normalize(mask):
    x, y, w, h = cv2.boundingRect(mask)
    out = np.zeros((48, 48), np.uint8)
    if not w or not h:
        return out
    scale = 40 / max(w, h)
    resized = cv2.resize(mask[y:y+h, x:x+w], (max(1, round(w*scale)), max(1, round(h*scale))),
                         interpolation=cv2.INTER_NEAREST)
    dy, dx = (48-resized.shape[0])//2, (48-resized.shape[1])//2
    out[dy:dy+resized.shape[0], dx:dx+resized.shape[1]] = resized
    return out


@lru_cache(maxsize=1)
def _banks():
    template_info()
    bank, hint_bank = [], []
    for index in range(1, 14):
        alpha = cv2.imread(str(ASSETS / f'shape-{index}.png'), cv2.IMREAD_UNCHANGED)[:, :, 3]
        alpha = cv2.copyMakeBorder(alpha, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=0)
        h, w = alpha.shape
        variants = []
        for angle in range(0, 360, 5):
            rotated = cv2.warpAffine(alpha, cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, 1), (w, h))
            binary = (rotated > 100).astype(np.uint8)
            bank.append(_normalize(binary).ravel())
            if angle % 10:
                continue
            x, y, rw, rh = cv2.boundingRect(binary)
            for size in range(18, 35, 2):
                resized = cv2.resize(rotated[y:y+rh, x:x+rw],
                                     (max(1, round(rw*size/max(rw, rh))), max(1, round(rh*size/max(rw, rh)))))
                variants.append((resized > 100).astype(np.float32))
        hint_bank.append(variants)
    bank = np.asarray(bank, np.float32)
    return bank, bank.sum(axis=1), hint_bank


def _classify(mask):
    bank, total, _ = _banks()
    vector = _normalize(mask).ravel().astype(np.float32)
    # 小矩阵不用 BLAS 线程池，避免线程启动开销主导每道题的耗时。
    intersection = np.einsum('ij,j->i', bank, vector, optimize=False)
    scores = intersection / np.maximum(1, total+vector.sum()-intersection)
    return scores.reshape(13, 72).max(axis=1)


def _fit_hint(mask, valid):
    """在线覆盖区域不奖不罚；在原尺寸搜索，避免裁掉遮挡边缘后改变形状。"""
    _, _, hint_bank = _banks()
    mask, valid = mask.astype(np.float32), valid.astype(np.float32)
    total, scores = mask.sum(), []
    for variants in hint_bank:
        best = 0.
        for kernel in variants:
            if kernel.shape[0] > mask.shape[0] or kernel.shape[1] > mask.shape[1]:
                continue
            intersection = cv2.matchTemplate(mask, kernel, cv2.TM_CCORR)
            visible = cv2.matchTemplate(valid, kernel, cv2.TM_CCORR)
            score = intersection / np.maximum(1, total+visible-intersection)
            best = max(best, float(score.max()))
        scores.append(best)
    return np.asarray(scores)


def _remove_specks(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for index in range(1, count):
        if stats[index, 4] < 3:
            mask[labels == index] = 0


def _hints(image):
    alpha = (image[:, :, 3] > 100).astype(np.uint8)
    lines = cv2.HoughLinesP(alpha*255, 1, np.pi/180, threshold=45, minLineLength=60, maxLineGap=12)
    occlusion = np.zeros_like(alpha)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(occlusion, (x1, y1), (x2, y2), 1, 3)
    loc = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    occupied = np.flatnonzero(loc.sum(axis=0) > 2)
    groups = np.split(occupied, np.where(np.diff(occupied) > 6)[0]+1)
    result = []
    for group in groups:
        if len(group) < 7 or loc[:, group[0]:group[-1]+1].sum() < 50:
            continue
        left, right = max(0, int(group[0])-3), min(image.shape[1], int(group[-1])+4)
        crop = image[:, left:right]
        pixels = crop[:, :, :3][(crop[:, :, 3] > 150) & (loc[:, left:right] > 0)]
        if len(pixels) < 25:
            continue
        colors, counts = np.unique(pixels, axis=0, return_counts=True)
        same = np.max(np.abs(crop[:, :, :3].astype(float)-colors[counts.argmax()]), axis=2) < 6
        valid = ((crop[:, :, 3] < 50) | same).astype(np.uint8)
        valid[occlusion[:, left:right] > 0] = 0
        mask = (same & (crop[:, :, 3] > 80) & (valid > 0)).astype(np.uint8)
        _remove_specks(mask)
        x, y, w, h = cv2.boundingRect(mask)
        if not w or not h:
            continue
        scores = _fit_hint(mask, valid)
        ranked = np.sort(scores)
        result.append({'box': [left+x, y, w, h], 'scores': scores, 'gap': float(ranked[-1]-ranked[-2])})
    return result


def _targets(image):
    rgb = image[:, :, :3]
    float_rgb = rgb.astype(float)
    edges = cv2.Canny(rgb, 60, 140)
    distance = cv2.distanceTransform(255-edges, cv2.DIST_L2, 3)
    colors, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    colors = colors[np.argsort(counts)[-120:][::-1]]
    quantized = (rgb//16).astype(np.int32)
    labels = quantized[:, :, 0]*256+quantized[:, :, 1]*16+quantized[:, :, 2]
    ids, counts = np.unique(labels, return_counts=True)
    extra = [np.median(rgb[labels == value], axis=0) for value in ids[np.argsort(counts)[-70:]]]
    candidates, used = [], []
    for color in np.vstack((colors, extra)):
        if any(np.max(np.abs(color-old)) < 14 for old in used):
            continue
        used.append(color)
        mask = (np.max(np.abs(float_rgb-color), axis=2) < 20).astype(np.uint8)
        if image.shape[2] == 4:
            mask[image[:, :, 3] < 80] = 0
        _remove_specks(mask)
        # 膨胀只用于聚合分离的花瓣；实际匹配不填空洞，也不连接花瓣间隙。
        grouped = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped)
        for index in range(1, count):
            x, y, w, h, _ = map(int, stats[index])
            part = mask[y:y+h, x:x+w]*(labels[y:y+h, x:x+w] == index).astype(np.uint8)
            xx, yy, ww, hh = cv2.boundingRect(part)
            if min(ww, hh) < 22 or max(ww, hh) > 64 or max(ww, hh)/min(ww, hh) > 1.8 or part.sum() < 70:
                continue
            box = [x+xx, y+yy, ww, hh]
            cropped = part[yy:yy+hh, xx:xx+ww]
            boundary = cv2.morphologyEx(cropped, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
            if np.mean(distance[y+yy:y+yy+hh, x+xx:x+xx+ww][boundary] < 2.5) < .55:
                continue
            if any(sum(abs(a-b) for a, b in zip(box, item['box'])) < 10 for item in candidates):
                continue
            scores = _classify(cropped)
            if scores.max() >= .58:
                candidates.append({'box': box, 'scores': scores})
    candidates.sort(key=lambda item: float(item['scores'].max()), reverse=True)
    selected = []
    for item in candidates:
        x, y, w, h = item['box']
        for previous in selected:
            px, py, pw, ph = previous['box']
            if abs(x+w/2-px-pw/2)+abs(y+h/2-py-ph/2) < 15:
                previous['scores'] = np.maximum(previous['scores'], item['scores'])
                break
        else:
            selected.append(item)
    return sorted(selected, key=lambda item: item['box'][0])


def click_gocaptcha_icons(background, hint):
    start = time.monotonic()
    detail = dict(template_info())

    def finish(usable=False, score=0., points=None, reason='图标分割不完整'):
        return Prediction('icon', bool(usable), float(score), points=points or [], reason=reason,
                          detail=detail, elapsed_ms=(time.monotonic()-start)*1000)

    if (background is None or hint is None or background.ndim != 3 or hint.ndim != 3
            or background.shape[2] not in (3, 4) or hint.shape[2] != 4
            or min(background.shape[:2]) < 64 or min(hint.shape[:2]) < 18
            or max(background.shape[:2]) > 1000 or max(hint.shape[:2]) > 600):
        return finish(reason='GoCaptcha 模式需要透明提示图及原尺寸题图')
    hints, targets = _hints(hint), _targets(background)
    expected = _hint_count(hint)
    detail.update(hint_boxes=[item['box'] for item in hints], target_boxes=[item['box'] for item in targets],
                  foreground_hint_count=expected, hint_margins=[item['gap'] for item in hints])
    if expected != len(hints) or not 2 <= len(hints) <= 3 or not len(hints) <= len(targets) <= 12:
        return finish()
    h = np.stack([item['scores'] for item in hints])
    t = np.stack([item['scores'] for item in targets])
    # 提示先独立定类，背景强匹配不能反过来将模糊提示改认成另一个类别。
    classes = h.argmax(axis=1)
    pair = np.log(np.maximum(t[:, classes].T, 1e-5))
    ranked = sorted(((sum(pair[i, j] for i, j in enumerate(p)), p)
                     for p in permutations(range(len(targets)), len(hints))), reverse=True)
    total, assignment = ranked[0]
    margin = float(total-ranked[1][0]) if len(ranked) > 1 else 1.
    scores = [[float(h[i, classes[i]]), float(t[j, classes[i]])] for i, j in enumerate(assignment)]
    relative = [float(t[j, classes[i]]/t[j].max()) for i, j in enumerate(assignment)]
    checks = {'hint_fit': all(a >= .72 for a, b in scores), 'target_fit': all(b >= .57 for a, b in scores),
              'hint_unique': all(item['gap'] > .008 for item in hints),
              'target_consistent': all(value >= .85 for value in relative), 'assignment_unique': margin > .04}
    detail.update(classes=(classes+1).tolist(), scores=scores, assignment=list(assignment), margin=margin,
                  target_relative_fit=relative, checks=checks)
    points = [[targets[j]['box'][0]+targets[j]['box'][2]/2, targets[j]['box'][1]+targets[j]['box'][3]/2]
              for j in assignment]
    usable = all(checks.values())
    return finish(usable, min(min(pair) for pair in scores), points,
                  '公开形状模板排列候选' if usable else '图形不匹配或排列存在歧义')
