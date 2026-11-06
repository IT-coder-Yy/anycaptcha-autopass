"""缺口识别：多算法并行 + 置信度投票。

输入都是"人眼能看到的东西"——题目背景图、滑块碎片图（可选原始干净背景）。
绝不读取接口里泄露的答案字段（那只用于事后算误差，见 scenarios/）。

算法：
  1. template_match  模板匹配（边缘图 + alpha 掩码）—— 主力
  2. fill_match      填充色匹配 + 残留纹理裁决 —— 主力（灰色填充式缺口，线上 tianai）
  3. shape_chamfer   剪影轮廓倒角匹配 —— 兜底（碎片图不是背景裁片时靠它）
  4. canny_contour   Canny 边缘 + 轮廓形状先验 —— 兜底
  5. col_projection  列向梯度投影 + 双边缘间距先验 —— 兜底（不依赖碎片图内容）
  6. pixel_diff      像素差分 —— 仅在拿得到干净原图时参战（票重最高）

两个让投票真正可靠的关键设计：
  * **y 带约束**：滑块只能水平移动，所以缺口与碎片在图中的行范围一致，
    只在这几行里搜 x，直接消灭大部分误峰；
  * **统一证据裁判**：各算法的原始置信度尺度不可比（边缘图上 0.3 的峰其实很强，
    而"面积契合度 0.9"可能毫无意义）。因此对每个候选统一重新打分再投票，
    证据取"轮廓压在真实边缘上的比例"与"填充色残差排名"的较大值 ——
    两种题面各有一种看得见的证据，只用一种会把另一种的正确答案判成没证据。
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from common.vote import Candidate, vote

MIN_EVIDENCE = 0.22  # 证据低于此值的候选直接弃权


@dataclass
class GapResult:
    x: float | None  # 原图坐标系：碎片应落到的 x（碎片左沿）
    y: float | None
    conf: float
    candidates: list[Candidate]
    vote_detail: dict[str, Any]
    debug_png_b64: str | None = None
    piece_box: tuple[int, int, int, int] | None = None  # 碎片在其画布内的 alpha 包围盒


# ------------------------------------------------------------------ 工具


def decode_image(data: bytes, with_alpha: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_UNCHANGED if with_alpha else cv2.IMREAD_COLOR
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
    if arr is None:
        raise ValueError("图片解码失败")
    return arr


def flatten_on_white(img: np.ndarray) -> np.ndarray:
    """RGBA → BGR，透明处填**白**。

    线上提示条（`templateImage`）是一张**透明底**的 PNG：字是精确的
    `(128,128,128)`（= Java 的 `Color.gray`，与出题器源码一致），其余 alpha=0。
    用 `IMREAD_COLOR` 解出来透明处会变成**黑**底灰字 —— 极性和页面上看到的
    （白底灰字）相反。CV 路凑巧不受影响（`normalize_glyph` 会按边框亮度自动反相），
    但模型是按"白底灰字"训的，喂黑底会直接崩。所以在解码处就统一到白底。
    """
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        out = img[:, :, :3].astype(np.float32) * a + 255.0 * (1.0 - a)
        return out.round().astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img[:, :, :3]


def alpha_mask(rgba: np.ndarray, thr: int = 24) -> np.ndarray:
    if rgba.ndim == 3 and rgba.shape[2] == 4:
        return (rgba[:, :, 3] > thr).astype(np.uint8) * 255
    g = cv2.cvtColor(rgba, cv2.COLOR_BGR2GRAY)
    return cv2.inRange(g, 8, 247)


def alpha_bbox(rgba: np.ndarray, thr: int = 24) -> tuple[int, int, int, int]:
    """碎片图里非透明区域的包围盒 (x, y, w, h)。"""
    a = alpha_mask(rgba, thr)
    ys, xs = np.where(a > 0)
    if len(xs) == 0:
        h, w = rgba.shape[:2]
        return (0, 0, w, h)
    return (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1))


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)


def _edges(img: np.ndarray, lo: int = 60, hi: int = 180) -> np.ndarray:
    g = cv2.GaussianBlur(_gray(img), (3, 3), 0)
    return cv2.Canny(g, lo, hi)


def _peak_quality(res: np.ndarray, loc: tuple[int, int], min_gap: int) -> float:
    """主峰相对次峰的显著性 0-1：峰越孤立越可信。"""
    flat = res.copy()
    x0 = max(0, loc[0] - min_gap)
    x1 = min(flat.shape[1], loc[0] + min_gap + 1)
    flat[:, x0:x1] = -1.0
    second = float(flat.max()) if flat.size else 0.0
    best = float(res[loc[1], loc[0]])
    if best <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - max(0.0, second) / best))


@dataclass
class _Prep:
    bg: np.ndarray
    bg_edge: np.ndarray
    bg_edge_dil: np.ndarray
    bg_dist: np.ndarray  # 到最近背景边缘的距离场（倒角匹配用，低阈值边缘）
    band: tuple[int, int]  # 搜索行范围 [y0, y1)
    pbox: tuple[int, int, int, int]
    sil: np.ndarray | None  # 碎片轮廓（裁剪后的 alpha 掩码）
    sil_edge: np.ndarray | None
    sil_thin: np.ndarray | None  # 单像素细轮廓（倒角匹配用，点数少一半）
    tpl_edge: np.ndarray | None
    tpl_bgr: np.ndarray | None  # 碎片彩色像素（裁剪后）
    sil_in: np.ndarray | None  # 内缩后的掩码：避开描边光晕，只留"填充体"
    fill_rmse: np.ndarray | None = None  # 掩码内颜色残差（全图 2D，惰性计算）
    fill_znc: np.ndarray | None = None  # 掩码内零均值归一化互相关（同上）


def _prepare(bg: np.ndarray, piece: np.ndarray | None,
             pbox: tuple[int, int, int, int]) -> _Prep:
    px, py, pw, ph = pbox
    bg_edge = _edges(bg)
    dil = cv2.dilate(bg_edge, np.ones((5, 5), np.uint8), iterations=1)
    # 距离场要的是"召回"：轮廓压在任何一条真实边缘上都算命中，所以用更低的阈值
    # 多留一些弱边（实测平均倒角距离 5.1px → 3.4px，谷底更深）。
    dist = cv2.distanceTransform(255 - _edges(bg, 40, 120), cv2.DIST_L2, 3)
    margin = int(max(6, ph * 0.14))
    y0 = max(0, py - margin)
    y1 = min(bg.shape[0], py + ph + margin)
    if y1 - y0 < ph + 2:  # 兜底：搜索整图
        y0, y1 = 0, bg.shape[0]

    sil = sil_edge = sil_thin = tpl_edge = tpl_bgr = sil_in = None
    if piece is not None:
        m = alpha_mask(piece)[py:py + ph, px:px + pw]
        sil = m
        sil_edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        sil_thin = cv2.Canny(m, 50, 150)
        # 描边那一圈是"高光"，两边的填充色都被它污染 —— 内缩 2px 只留填充体
        sil_in = cv2.erode(m, np.ones((5, 5), np.uint8))
        tpl = piece[py:py + ph, px:px + pw]
        tpl_bgr = np.ascontiguousarray(tpl[:, :, :3])
        te = _edges(tpl)
        tpl_edge = cv2.bitwise_and(te, m)
        # 轮廓边缘并入模板：缺口最强的可见特征就是那圈描边
        tpl_edge = cv2.bitwise_or(tpl_edge, sil_edge)
    return _Prep(bg, bg_edge, dil, dist, (y0, y1), pbox, sil, sil_edge, sil_thin,
                 tpl_edge, tpl_bgr, sil_in)


def _edge_evidence(prep: _Prep, xi: int, yi: int) -> float:
    """碎片轮廓落在背景边缘上的比例（0-1）。"""
    px, py, pw, ph = prep.pbox
    win = prep.bg_edge_dil[yi:yi + ph, xi:xi + pw]
    sel = prep.sil_edge > 0
    n = int(sel.sum())
    if n == 0:
        return 0.0
    return float((win[sel] > 0).sum()) / n


def _fill_evidence(prep: _Prep, xi: int, yi: int) -> float:
    """填充色证据：这一处的掩码内颜色残差，在整条剖面里排到多好（0-1）。

    与边缘证据互补。线上"灰填充缺口"那一档，缺口边界是柔和的描边光晕，
    Canny 常常根本看不见，边缘证据会把正确答案判成没证据；反过来，
    本地题面的缺口是挖空+压暗，填充色未必对得上，那时靠边缘证据。
    所以两者取大 —— 满足任何一条都算"人眼能看见的证据"。
    """
    prof = _fill_profiles(prep)
    if prof is None:
        return 0.0
    rmse2d = prof[0]
    if not (0 <= yi < rmse2d.shape[0] and 0 <= xi < rmse2d.shape[1]):
        return 0.0
    row = rmse2d[yi]
    lo = float(row.min())
    mid = float(np.median(row))
    if mid - lo < 1e-6:
        return 0.0
    return float(np.clip(1.0 - (row[xi] - lo) / (mid - lo), 0.0, 1.0))


TEX_INFORMATIVE = 0.15  # 行内最佳纹理相关低于此值，就当"这题看不出纹理"，不做否决


def _texture_factor(prep: _Prep, xi: int, yi: int) -> float:
    """残留纹理否决系数（0.15-1.0）：这一处到底像不像"碎片原来所在的位置"。

    形状类算法（模板匹配、倒角）天生分不清真缺口和**同形状的干扰块** ——
    形状一模一样，它们会一起投给干扰块，把唯一能分辨的算法票压下去
    （实测线上一张图里出现过 3 条一样的鱼、2 个一样的圆）。

    能分辨的只有一件事：碎片的像素来自真缺口那一处，所以只有在那里，
    题图残留的纹理才和碎片的纹理同相位。干扰块是另贴的一块，不相关。

    只有当这一行确实存在明显的纹理相关时才启用否决，否则（纯色填充、
    大片平坦背景）返回 1.0 不干预 —— 没信息时不该假装有信息。
    """
    prof = _fill_profiles(prep)
    if prof is None:
        return 1.0
    znc2d = prof[1]
    if not (0 <= yi < znc2d.shape[0] and 0 <= xi < znc2d.shape[1]):
        return 1.0
    row = znc2d[yi]
    z_max = float(row.max())
    if z_max <= TEX_INFORMATIVE:
        return 1.0
    return float(np.clip(0.30 + 0.70 * float(row[xi]) / z_max, 0.15, 1.0))


def _evidence(prep: _Prep, x: float, y: float | None) -> float:
    """统一证据：尺度可比、不易被欺骗的"人眼可见证据"打分（0-1）。"""
    if prep.sil_edge is None:
        return 0.5  # 无碎片图时无法验证，给中性分
    px, py, pw, ph = prep.pbox
    xi = int(round(x))
    yi = int(round(y if y is not None else py))
    h, w = prep.bg_edge_dil.shape[:2]
    if xi < 0 or yi < 0 or xi + pw > w or yi + ph > h:
        return 0.0
    seen = max(_edge_evidence(prep, xi, yi), _fill_evidence(prep, xi, yi))
    return seen * _texture_factor(prep, xi, yi)


def _calibrate(cand: Candidate, prep: _Prep) -> Candidate:
    """用统一证据校准置信度；证据过弱直接弃权。"""
    if cand.x is None:
        return cand
    ev = _evidence(prep, cand.x, cand.y)
    cand.extra["evidence"] = round(ev, 4)
    if ev < MIN_EVIDENCE:
        cand.ok = False
        cand.note = (cand.note + f" | 证据不足 {ev:.2f}，弃权").strip(" |")
        cand.conf = 0.0
        return cand
    raw = max(0.0, min(1.0, cand.conf))
    cand.conf = round(0.30 * raw + 0.70 * ev, 4)
    cand.note = (cand.note + f" | 证据 {ev:.2f}").strip(" |")
    return cand


# ------------------------------------------------------------------ 算法 1


def algo_template_match(prep: _Prep) -> Candidate:
    t0 = time.perf_counter()
    px, py, pw, ph = prep.pbox
    if prep.tpl_edge is None:
        return Candidate("template_match", None, ok=False, weight=1.0, note="无碎片图")
    y0, y1 = prep.band
    band = prep.bg_edge[y0:y1]
    if band.shape[0] < ph or band.shape[1] < pw:
        return Candidate("template_match", None, ok=False, weight=1.0, note="搜索带过小",
                         ms=(time.perf_counter() - t0) * 1000)
    res = cv2.matchTemplate(band, prep.tpl_edge, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    q = _peak_quality(res, loc, max(8, pw // 2))
    # 二值边缘图上 0.3 已是强峰：用较陡的映射，别把好峰压死
    raw = max(0.0, min(1.0, mx / 0.45)) * (0.45 + 0.55 * q)
    return Candidate("template_match", float(loc[0]), float(y0 + loc[1]), conf=raw, weight=1.0,
                     ms=(time.perf_counter() - t0) * 1000,
                     note=f"峰值{mx:.3f} 显著性{q:.2f}",
                     extra={"box": [int(loc[0]), int(y0 + loc[1]), int(pw), int(ph)]})


# ------------------------------------------------------------------ 算法 2


FILL_TEX_W = 2.0      # 残留纹理项的权重：它是唯一能识破同形状干扰块的证据
FLAT_SD = 4.0         # 灰度标准差低于此值就算"这块是平的"，相关系数无意义
FILL_SCORE_REF = 1.6  # 得分到这个量级就算"证据充分"


def _fill_profiles(prep: _Prep) -> tuple[np.ndarray, np.ndarray] | None:
    """惰性算两张全图响应：掩码内颜色 RMSE、掩码内零均值归一化互相关。"""
    if prep.tpl_bgr is None or prep.sil_in is None:
        return None
    if prep.fill_rmse is None:
        m = prep.sil_in
        n = float((m > 0).sum())
        if n < 40:
            return None
        # 颜色残差：碎片是什么色，缺口就该是什么色（同一套灰混合）
        sq = cv2.matchTemplate(prep.bg, prep.tpl_bgr, cv2.TM_SQDIFF,
                               mask=cv2.merge([m] * 3))
        prep.fill_rmse = np.sqrt(np.maximum(sq, 0.0) / (n * 3.0))
        # 残留纹理：填充是半透明的，缺口里还留着原画的影子；干扰块是纯色，没有
        g = _gray(prep.bg).astype(np.float32)
        tg = cv2.cvtColor(prep.tpl_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mf = (m > 0).astype(np.float32)
        tm = float((tg * mf).sum()) / n
        t_sd = float(np.sqrt((((tg - tm) ** 2) * mf).sum() / n))
        num = cv2.matchTemplate(g, (tg - tm) * mf, cv2.TM_CCORR)
        s1 = cv2.matchTemplate(g, mf, cv2.TM_CCORR)
        s2 = cv2.matchTemplate(g * g, mf, cv2.TM_CCORR)
        mu = s1 / n
        sd = np.sqrt(np.maximum(s2 / n - mu * mu, 0.0))
        if t_sd < FLAT_SD:
            # 碎片本身没纹理，就没有"纹理是否同相位"这回事
            prep.fill_znc = np.zeros_like(num)
        else:
            znc = num / (n * np.maximum(sd, FLAT_SD) * t_sd)
            # 窗口本身是平的（大片白/天空）时相关系数没有意义，直接判 0，
            # 否则 sd→0 会把噪声放大成"完美相关"（实测出现过 +13.6 的荒唐值）
            prep.fill_znc = np.where(sd < FLAT_SD, 0.0, np.clip(znc, -1.0, 1.0))
    return prep.fill_rmse, prep.fill_znc


def algo_fill_match(prep: _Prep) -> Candidate:
    """填充色匹配 + 残留纹理裁决 —— 线上 tianai 这类"灰色填充缺口"的主力。

    这一档题面里，缺口不是"挖空"，而是把碎片形状**按同一套灰色混合填在原图上**，
    碎片本身也是那个混合结果。所以最直接的证据就是人眼在做的事：
    找一块形状吻合、**颜色也吻合**的填充区（掩码内 RMSE 最小）。

    麻烦在于站点会画**同形状的干扰块**（实测一张图里出现过 3 条一样的鱼、
    2 个一样的圆），只比颜色分不出真假。真缺口是半透明混合，里面还留着原画的
    影子；干扰块是纯色，没有影子。于是 RMSE 并列时用"残留纹理相关"裁决 ——
    实测 c03 三条鱼里 RMSE 最优的那条是假的（ZNCC −0.11），真缺口 ZNCC +0.19。

    边缘类算法在这一档会被密集纹理骗：题图里树冠、电线到处是边，
    "轮廓压在边上"随手就能满足（见 backend/probe_out/tianai/slider.md）。
    """
    t0 = time.perf_counter()
    prof = _fill_profiles(prep)
    px, py, pw, ph = prep.pbox
    if prof is None:
        return Candidate("fill_match", None, ok=False, weight=1.4, note="无碎片图",
                         ms=(time.perf_counter() - t0) * 1000)
    rmse2d, znc2d = prof
    y0, y1 = prep.band
    r0, r1 = y0, min(rmse2d.shape[0], max(y0 + 1, y1 - ph + 1))
    band_rmse = rmse2d[r0:r1]
    band_znc = znc2d[r0:r1]
    if band_rmse.size == 0:
        return Candidate("fill_match", None, ok=False, weight=1.4, note="搜索带过小",
                         ms=(time.perf_counter() - t0) * 1000)

    # 打分：填充色排名 + 残留纹理。纹理项权重更高 —— 它是唯一能区分
    # "真缺口"和"同形状干扰块"的证据，而填充色对两者一样好。
    # 两项必须在**同一行**上取，所以整条带一起打分再取列最优，
    # 不能先按 RMSE 挑行、再去那一行读纹理（会读到另一行的纹理）。
    med = float(np.median(band_rmse))
    fit2d = np.maximum(0.0, 1.0 - band_rmse / max(med, 1e-6))
    score2d = fit2d + FILL_TEX_W * np.maximum(0.0, band_znc)
    rows = np.argmax(score2d, axis=0)
    cols = np.arange(score2d.shape[1])
    prof_s = score2d[rows, cols]
    prof_r = band_rmse[rows, cols]
    prof_z = band_znc[rows, cols]

    # 局部极大 → 候选清单（同一个峰只留最高的一点）
    guard = max(8, pw // 2)
    cands: list[int] = []
    for i in np.argsort(-prof_s):
        if all(abs(int(i) - c) > guard for c in cands):
            cands.append(int(i))
        if len(cands) >= 5:
            break
    if not cands:
        return Candidate("fill_match", None, ok=False, weight=1.4, note="无候选",
                         ms=(time.perf_counter() - t0) * 1000)

    best = cands[0]
    rival = prof_s[cands[1]] if len(cands) > 1 else 0.0
    note = (f"RMSE{prof_r[best]:.0f}(中位{med:.0f}) 残留纹理{prof_z[best]:+.2f} "
            f"得分{prof_s[best]:.2f} 次优{rival:.2f} 候选{len(cands)}")

    # 亚像素：在 RMSE 谷底做抛物线插值（得分曲线含纹理项，谷形不如 RMSE 干净）
    x_sub = float(best)
    if 0 < best < len(prof_r) - 1:
        a, b, c = prof_r[best - 1], prof_r[best], prof_r[best + 1]
        den = a - 2 * b + c
        if den > 1e-6:
            x_sub = best + float(np.clip(0.5 * (a - c) / den, -1.0, 1.0))

    margin = max(0.0, min(1.0, (prof_s[best] - rival) / max(prof_s[best], 1e-6)))
    raw = max(0.0, min(1.0, prof_s[best] / FILL_SCORE_REF)) * (0.45 + 0.55 * margin)
    return Candidate("fill_match", x_sub, float(r0 + rows[best]), conf=raw, weight=1.4,
                     ms=(time.perf_counter() - t0) * 1000, note=note,
                     extra={"box": [int(round(x_sub)), int(r0 + rows[best]), int(pw), int(ph)],
                            "rmse": round(float(prof_r[best]), 2),
                            "znc": round(float(prof_z[best]), 3),
                            "cands": len(cands)})


# ------------------------------------------------------------------ 算法 3


CHAMFER_REF_PX = 6.0  # 平均倒角距离达到这个量级就基本没信息了


def algo_shape_chamfer(prep: _Prep) -> Candidate:
    """剪影轮廓倒角匹配：碎片 alpha 的那圈轮廓，应该逐点压在题图的边缘上。

    专治"碎片图不是背景裁片"的题面 —— tianai 线上就是这样：碎片是一块**纯灰剪影**，
    缺口也是同形状的纯灰剪影。灰对灰做灰度/边缘相关很弱（实测峰值只有 0.2 上下），
    但两者的轮廓完全同形，改成"轮廓点到最近边缘的平均距离"就分得很开。

    实现上按行切片累加距离场，避免逐点 gather：336 个轮廓点 × 500 个候选 x
    也就是 336 次长度 500 的向量加法（整题多花 ~20ms）。
    """
    t0 = time.perf_counter()
    px, py, pw, ph = prep.pbox
    if prep.sil_thin is None:
        return Candidate("shape_chamfer", None, ok=False, weight=1.1, note="无碎片图",
                         ms=(time.perf_counter() - t0) * 1000)
    pts = np.column_stack(np.nonzero(prep.sil_thin))  # (dy, dx)
    y0, y1 = prep.band
    n_x = prep.bg.shape[1] - pw
    if len(pts) < 24 or n_x <= 1:
        return Candidate("shape_chamfer", None, ok=False, weight=1.1, note="轮廓点太少",
                         ms=(time.perf_counter() - t0) * 1000)

    d = prep.bg_dist
    best = (float("inf"), 0, py)
    prof_best: np.ndarray | None = None
    for yy in range(y0, max(y0 + 1, y1 - ph + 1)):
        prof = np.zeros(n_x, np.float32)
        for dy, dx in pts:
            prof += d[yy + dy, dx:dx + n_x]
        prof /= len(pts)
        i = int(np.argmin(prof))
        if prof[i] < best[0]:
            best, prof_best = (float(prof[i]), i, yy), prof
    mean_d, xi, yy = best
    prof = prof_best if prof_best is not None else np.zeros(n_x, np.float32)

    # 次优谷：主谷之外的最小值，用来判断这个谷是不是唯一
    guard = max(8, pw // 2)
    masked = prof.copy()
    masked[max(0, xi - guard):min(n_x, xi + guard + 1)] = np.inf
    second = float(masked.min()) if np.isfinite(masked).any() else mean_d * 2
    q = max(0.0, min(1.0, (second - mean_d) / max(second, 1e-6)))
    raw = max(0.0, min(1.0, 1.0 - mean_d / CHAMFER_REF_PX)) * (0.45 + 0.55 * q)

    # 抛物线插值取亚像素谷底
    x_sub = float(xi)
    if 0 < xi < n_x - 1:
        a, b, c = prof[xi - 1], prof[xi], prof[xi + 1]
        den = a - 2 * b + c
        if den > 1e-6:
            x_sub = xi + float(np.clip(0.5 * (a - c) / den, -1.0, 1.0))
    return Candidate("shape_chamfer", x_sub, float(yy), conf=raw, weight=1.1,
                     ms=(time.perf_counter() - t0) * 1000,
                     note=f"平均倒角{mean_d:.2f}px 次优{second:.2f} 唯一性{q:.2f}",
                     extra={"box": [int(round(x_sub)), int(yy), int(pw), int(ph)],
                            "chamfer_px": round(mean_d, 3)})


# ------------------------------------------------------------------ 算法 4


def algo_canny_contour(prep: _Prep) -> Candidate:
    t0 = time.perf_counter()
    px, py, pw, ph = prep.pbox
    y0, y1 = prep.band
    e = prep.bg_edge[y0:y1].copy()
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_area, target_ratio = pw * ph, pw / max(1, ph)
    best: tuple[float, tuple[int, int, int, int]] | None = None
    n_cand = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < pw * 0.55 or h < ph * 0.55 or w > pw * 1.8 or h > ph * 1.8:
            continue
        if x < pw * 0.5:  # 最左侧是碎片自身所在位置，排除
            continue
        n_cand += 1
        s_area = 1.0 - min(1.0, abs(w * h - target_area) / target_area)
        ratio = w / max(1, h)
        s_ratio = 1.0 - min(1.0, abs(ratio - target_ratio) / max(0.2, target_ratio))
        score = 0.6 * s_area + 0.4 * s_ratio
        if best is None or score > best[0]:
            best = (score, (x, y0 + y, w, h))
    ms = (time.perf_counter() - t0) * 1000
    if best is None:
        return Candidate("canny_contour", None, ok=False, weight=0.7, ms=ms, note="无候选轮廓")
    score, (x, y, w, h) = best
    # 候选越多，"形状契合"本身越不可信 —— 原始置信度按候选数打折
    raw = max(0.0, min(1.0, score)) / (1.0 + 0.35 * max(0, n_cand - 1))
    return Candidate("canny_contour", float(x), float(y), conf=raw, weight=0.7, ms=ms,
                     note=f"轮廓{w}x{h} 契合{score:.2f} 候选{n_cand}",
                     extra={"box": [x, y, w, h]})


# ------------------------------------------------------------------ 算法 5


def algo_col_projection(prep: _Prep) -> Candidate:
    """列向梯度投影：缺口的左右两条竖边间距 ≈ 碎片宽度。不依赖碎片图内容。"""
    t0 = time.perf_counter()
    px, py, pw, ph = prep.pbox
    y0, y1 = prep.band
    g = cv2.GaussianBlur(_gray(prep.bg)[y0:y1], (5, 5), 0)
    prof = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)).sum(axis=0)
    k = max(3, pw // 12) | 1
    prof = cv2.GaussianBlur(prof.reshape(1, -1), (k, 1), 0).ravel()
    if prof.max() <= 0:
        return Candidate("col_projection", None, ok=False, weight=0.6, note="无梯度",
                         ms=(time.perf_counter() - t0) * 1000)
    prof = prof / prof.max()
    lo, hi = int(pw * 0.6), len(prof) - pw - 2
    if hi <= lo:
        return Candidate("col_projection", None, ok=False, weight=0.6, note="图宽不足",
                         ms=(time.perf_counter() - t0) * 1000)
    xs = np.arange(lo, hi)
    inner = np.array([prof[x + 3:x + pw - 3].mean() if pw > 8 else 0.0 for x in xs])
    score = prof[xs] + prof[xs + pw - 1] - 0.8 * inner
    i = int(np.argmax(score))
    best_x, s = int(xs[i]), float(score[i])
    rival = np.sort(score)[-max(2, len(score) // 50):]
    q = 1.0 - float(np.mean(rival[:-1])) / s if s > 0 else 0.0
    raw = max(0.0, min(1.0, s / 2.0)) * (0.5 + 0.5 * max(0.0, min(1.0, q)))
    return Candidate("col_projection", float(best_x), float(py), conf=raw, weight=0.6,
                     ms=(time.perf_counter() - t0) * 1000, note=f"双边缘得分{s:.2f}")


# ------------------------------------------------------------------ 算法 6


def algo_pixel_diff(prep: _Prep, ref: np.ndarray) -> Candidate:
    """像素差分 —— 仅当拿得到"未挖缺口的干净原图"时可用。"""
    t0 = time.perf_counter()
    px, py, pw, ph = prep.pbox
    bg = prep.bg
    if ref.shape[:2] != bg.shape[:2]:
        ref = cv2.resize(ref, (bg.shape[1], bg.shape[0]))
    d = cv2.absdiff(_gray(bg), _gray(ref))
    d = cv2.GaussianBlur(d, (5, 5), 0)
    _, m = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ms = (time.perf_counter() - t0) * 1000
    if not cnts:
        return Candidate("pixel_diff", None, ok=False, weight=1.2, ms=ms, note="差分无响应")
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    fit = 1.0 - min(1.0, (abs(w - pw) / max(1, pw) + abs(h - ph) / max(1, ph)) / 2)
    return Candidate("pixel_diff", float(x), float(y), conf=max(0.3, min(1.0, 0.6 + 0.4 * fit)),
                     weight=1.2, ms=ms, note=f"差分区{w}x{h}", extra={"box": [x, y, w, h]})


# ------------------------------------------------------------------ 编排


def detect_gap(
    bg_bytes: bytes,
    piece_bytes: bytes | None,
    ref_bytes: bytes | None = None,
    tolerance: float = 12.0,
    want_debug: bool = True,
) -> GapResult:
    bg = decode_image(bg_bytes)
    if piece_bytes:
        piece = decode_image(piece_bytes, with_alpha=True)
        pbox = alpha_bbox(piece)
    else:
        piece = None
        side = int(bg.shape[0] * 0.28)
        pbox = (0, (bg.shape[0] - side) // 2, side, side)

    prep = _prepare(bg, piece, pbox)
    cands: list[Candidate] = []
    if piece is not None:
        cands.append(algo_template_match(prep))
        cands.append(algo_fill_match(prep))
        cands.append(algo_shape_chamfer(prep))
    cands.append(algo_canny_contour(prep))
    cands.append(algo_col_projection(prep))
    if ref_bytes:
        cands.append(algo_pixel_diff(prep, decode_image(ref_bytes)))

    cands = [_calibrate(c, prep) for c in cands]
    v = vote(cands, tolerance=tolerance)
    debug = _draw_debug(bg, cands, v, pbox, prep.band) if want_debug else None
    return GapResult(x=v["x"], y=v["y"], conf=v["conf"], candidates=cands,
                     vote_detail=v, debug_png_b64=debug, piece_box=pbox)


_COLORS = {
    "template_match": (0, 220, 90),
    "fill_match": (255, 255, 0),
    "shape_chamfer": (0, 160, 255),
    "canny_contour": (255, 170, 0),
    "col_projection": (255, 80, 200),
    "pixel_diff": (80, 200, 255),
}


def _draw_debug(bg: np.ndarray, cands: list[Candidate], v: dict, pbox, band) -> str:
    img = bg[:, :, :3].copy()
    px, py, pw, ph = pbox
    y0, y1 = band
    over = img.copy()
    cv2.rectangle(over, (0, y0), (img.shape[1], y1), (60, 60, 60), -1)
    img = cv2.addWeighted(img, 0.82, over, 0.18, 0)
    cv2.putText(img, "y-band", (4, max(11, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 200, 200), 1, cv2.LINE_AA)
    for c in cands:
        if c.x is None:
            continue
        col = _COLORS.get(c.name, (200, 200, 200))
        if not c.ok:
            col = tuple(int(v0 * 0.35) for v0 in col)
        box = c.extra.get("box")
        x, y, w, h = box if box else (int(c.x), int(c.y if c.y is not None else py), pw, ph)
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
        cv2.putText(img, f"{c.name[:4]} {c.conf:.2f}", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    if v["x"] is not None:
        x = int(v["x"])
        cv2.line(img, (x, 0), (x, img.shape[0]), (0, 0, 255), 2)
        cv2.putText(img, f"VOTE x={x} conf={v['conf']:.2f}", (max(2, x - 90), img.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""
