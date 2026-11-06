"""ROTATE（旋转验证）：求"把碎片顺时针转多少度才对齐"。

## 题面结构决定了判据

出题器（`app/mock/tianai_clone.py::render_rotate` 里有逐行复刻与出处）做了两件事：

- 背景正中被画上一个**黑色圆盘**（`fixed.png` 的 BGR 全 0、alpha 是实心圆盘），
  所以背景在 `r <= R` 内**没有内容**，可用的照片只在 `r > R`；
- 碎片是背景**原始**的中心圆盘，被顺时针转过 `degree`，边缘再叠一圈
  半透明装饰环（`active.png`，半径 0.861R~1.04R）。

于是唯一的判据是**盘边界处内容连续**：碎片在 `r` 略小于 R 处的内容，
应当接上背景在 `r` 略大于 R 处的内容。旋转在极坐标里就是角向的**循环平移**，
所以这题和 `align.py` 的接缝定位是同一台机器 —— 一维循环互相关。

两条必须绕开的污染：

1. 装饰环占掉 `r/R ∈ [0.861, 1.0]`，所以碎片侧只能用 `r/R < 0.845`；
2. 黑圆盘边缘有抗锯齿过渡，背景侧要从 `r/R > 1.01` 起用。

中间空出的这一段半径（0.845R → 1.01R，约 0.165R）是这个方法的**固有代价**：
两侧比的不是同一圈半径的内容。补法是**多组半径配对各算一条相关曲线再叠加**，
让不同半径尺度互相投票，而不是赌某一圈。

## 旋转方向

出题器用 `graphics2d.rotate(+degree, w/2, h/2)`：Java2D 的 y 轴向下，正角度是
**顺时针**。本模块对外的 `deg` 一律是"**把碎片顺时针再转 deg 度就对齐**"，
与前端 `rotate.js` 的 `transform: rotate(+deg)`（CSS 正角度也是顺时针）同向，
所以场景层拿到 `deg` 可以直接换成把手位移：`moveX = deg * end / 360`。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from common.vote import Candidate

# 角向采样数。1 个格子 = 0.5°，而站点容错是 ±7.2°（见 tianai_clone.ROTATE_TOLERANT_DEG），
# 所以分辨率远远够用；取偶数便于 FFT。
N_ANG = 720

# 两侧的采样半径（相对 R）。
#
# 第一版把碎片侧取到 0.845R 以内（避开装饰环）、背景侧从 1.015R 起，结果只有
# 32.5% 命中（`tools/bench_rotate.py` 实测）。根因不是方法错，是**两侧比的不是同一圈**：
# 0.8R 与 1.2R 处的照片内容本来就不同，相关峰自然糊。
#
# 改法：贴着边界取，让配对的半径尽量接近。代价是碎片侧会吃到装饰环
# （0.861R~1.04R，半透明）。这个代价可以被吃掉，因为签名是**逐半径去均值去尺度**的 ——
# 环把对比度压低，归一化会补回来；环自身的角向纹理跟着碎片转，与背景无关，只是噪声。
PIECE_R = (0.72, 0.985)
BG_R = (1.005, 1.27)
N_RAD = 16
# 半径差的加权尺度（相对 R）：越接近的一对权重越高。σ 取 0.08R —— 再小就只剩边界
# 那一圈、抗噪变差；再大就退回第一版"跨半径乱配"的毛病。
RADIUS_SIGMA = 0.08

# 置信度标定见 tools/bench_rotate.py：把"相关峰的显著性"压成 0-1
PEAK_SCALE = 6.0

# 换题闸门。口径与点选那一档一致：被拦下不是失败，是**换一道新题重来**
# （代价一次 generate，不发 verify），所以最大化的是
#     P(成功) = [1 - (1-放行)^(R+1)] x (1 - 闸门后错判)，R = MAX_REFRESH = 3
# 而不是"放行率 x 准确率"（那个口径永远得出"别设闸门"，理由见 probe_out/model.md §2）。
#
# `tools/bench_rotate.py 150` 实测（达标线 = 站点容错 ±7.2°）：
#
#   闸门   程序底图 放行/错判/P    真站底图 放行/错判/P
#   0.00     1.000 / 0.100 / 0.900   1.000 / 0.047 / 0.953
#   0.60     0.953 / 0.063 / 0.937   0.940 / 0.028 / 0.972
#   0.70     0.900 / 0.037 / 0.963   0.860 / 0.016 / 0.984
#   0.75     0.793 / 0.017 / 0.981   0.773 / 0.000 / 0.997   <- 两条曲线的共同峰值
#   0.80     0.780 / 0.017 / 0.981   0.713 / 0.000 / 0.993
#   0.85     0.767 / 0.017 / 0.980   0.667 / 0.000 / 0.988
ROTATE_GATE = 0.75


@dataclass
class RotateResult:
    deg: float | None
    conf: float = 0.0
    candidates: list[Candidate] = field(default_factory=list)
    curve: list[float] = field(default_factory=list)
    radius: float = 0.0
    ms: float = 0.0
    note: str = ""
    debug: dict[str, Any] = field(default_factory=dict)


def piece_geometry(piece_bgra: np.ndarray) -> tuple[float, float, float]:
    """从碎片的 alpha 反推 (盘心 x, 盘心 y, 半径)。

    不写死 `(tpl_w/2, H/2)`：线上模板尺寸随 `templateImageTag` 变，
    而 alpha 本身就是圆盘，量它比信配置可靠。
    """
    a = piece_bgra[:, :, 3]
    m = (a > 100).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        h, w = a.shape
        return (w - 1) / 2.0, (h - 1) / 2.0, min(w, h) / 2.0
    cx = float(xs.mean())
    cy = float(ys.mean())
    r = float(np.sqrt(m.sum() / np.pi))
    return cx, cy, r


def _polar(img: np.ndarray, cx: float, cy: float,
           r0: float, r1: float, n_rad: int = N_RAD) -> tuple[np.ndarray, np.ndarray]:
    """按 (角度, 半径) 采样，返回 (`n_rad x N_ANG x C` 的样本, 各半径值)。

    越界的半径整圈丢掉（返回值里那一圈全是 NaN，由 `_profiles` 过滤）。
    """
    if img.ndim == 2:
        img = img[:, :, None]
    ang = np.linspace(0.0, 2.0 * np.pi, N_ANG, endpoint=False, dtype=np.float32)
    rad = np.linspace(r0, r1, n_rad, dtype=np.float32)
    xs = (cx + np.cos(ang)[None, :] * rad[:, None]).astype(np.float32)
    ys = (cy + np.sin(ang)[None, :] * rad[:, None]).astype(np.float32)
    h, w = img.shape[:2]
    out = cv2.remap(img.astype(np.float32), xs, ys, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    if out.ndim == 2:
        out = out[:, :, None]
    bad = (xs < 0) | (xs > w - 1) | (ys < 0) | (ys > h - 1)
    out[bad.any(axis=1)] = np.nan
    return out, rad


def _profiles(polar: np.ndarray, rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐半径逐通道归一化成单位向量，返回 (剩下的 profile 堆, 对应半径)。

    逐半径归一化是必须的：整幅图的径向明暗梯度会盖住角向结构，
    而径向梯度在两侧（碎片内侧 / 背景外侧）本来就不一样。
    顺带把装饰环压低的对比度补回来。
    """
    keep_p: list[np.ndarray] = []
    keep_r: list[float] = []
    for i in range(polar.shape[0]):
        if np.isnan(polar[i]).any():
            continue
        chans = []
        for c in range(polar.shape[2]):
            v = polar[i, :, c]
            v = v - v.mean()
            n = float(np.linalg.norm(v))
            if n < 1e-6:
                continue
            chans.append(v / n)
        if not chans:
            continue
        keep_p.append(np.stack(chans, axis=0))
        keep_r.append(float(rad[i]))
    if not keep_p:
        return np.zeros((0, 0, N_ANG), np.float32), np.zeros(0, np.float32)
    return np.stack(keep_p, axis=0), np.asarray(keep_r, np.float32)


def _circ_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """循环互相关：返回长度 N_ANG 的曲线，`out[k]` = 把 b 循环右移 k 后与 a 的内积。"""
    fa = np.fft.rfft(a)
    fb = np.fft.rfft(b)
    return np.fft.irfft(fa * np.conj(fb), n=len(a))


def _parabolic(curve: np.ndarray, k: int) -> float:
    """峰顶抛物线插值，拿亚格精度（1 格 = 360/N_ANG 度）。"""
    n = len(curve)
    y0, y1, y2 = curve[(k - 1) % n], curve[k], curve[(k + 1) % n]
    d = y0 - 2.0 * y1 + y2
    if abs(d) < 1e-12:
        return float(k)
    return float(k) + 0.5 * (y0 - y2) / d


def _peak(curve: np.ndarray) -> tuple[float, float]:
    """返回 (亚格峰位, 显著性)。显著性 = (峰 - 中位) / 标准差。"""
    k = int(np.argmax(curve))
    pos = _parabolic(curve, k)
    med = float(np.median(curve))
    sd = float(curve.std())
    sig = (float(curve[k]) - med) / sd if sd > 1e-12 else 0.0
    return pos % len(curve), sig


def _pair_curve(bg_sig: np.ndarray, pc_sig: np.ndarray) -> np.ndarray:
    return _circ_corr(bg_sig, pc_sig)


def disc_center(bg_bgr: np.ndarray, r_hint: float) -> tuple[float, float]:
    """从背景那块**纯黑圆盘**量出盘心，而不是信"它一定在画面正中"。

    实测（3 道真题）：盘内灰度均值 0.2~0.3、标准差 0.7~0.9，盘外均值 146~208、
    标准差 32~90 —— 黑得非常干净，阈值取 12 就能分开。
    量不出合理结果（面积对不上 πr²）时退回画面中心，那是出题器写死的位置。
    """
    h, w = bg_bgr.shape[:2]
    fallback = ((w - 1) / 2.0, (h - 1) / 2.0)
    g = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
    m = (g <= 12).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    want = np.pi * r_hint * r_hint
    best, best_d = None, 1e18
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        d = abs(area - want) / want
        if d < best_d:
            best, best_d = i, d
    if best is None or best_d > 0.25:
        return fallback
    return float(cent[best][0]), float(cent[best][1])


def algo_polar_corr(bg_bgr: np.ndarray, piece_bgra: np.ndarray,
                    use_gradient: bool = False) -> tuple[Candidate, np.ndarray]:
    """跨边界的角向循环互相关，按半径差加权把所有配对叠起来。

    `use_gradient=True` 时先对每条 profile 做角向一阶差分 —— 那一版强调边缘、
    对低频明暗差不敏感，与原始 profile 互为独立判据（进投票）。
    """
    t0 = time.perf_counter()
    name = "polar_grad" if use_gradient else "polar_corr"
    pcx, pcy, R = piece_geometry(piece_bgra)
    bgx, bgy = disc_center(bg_bgr, R)

    def prep(stack: np.ndarray) -> np.ndarray:
        if not use_gradient or stack.size == 0:
            return stack
        g = np.diff(stack, axis=-1, append=stack[..., :1])
        g -= g.mean(axis=-1, keepdims=True)
        n = np.linalg.norm(g, axis=-1, keepdims=True)
        return np.divide(g, np.maximum(n, 1e-9))

    pp, pr = _profiles(*_polar(piece_bgra[:, :, :3], pcx, pcy, PIECE_R[0] * R, PIECE_R[1] * R))
    bp, br = _profiles(*_polar(bg_bgr, bgx, bgy, BG_R[0] * R, BG_R[1] * R))
    ms0 = (time.perf_counter() - t0) * 1000.0
    if pp.size == 0 or bp.size == 0:
        return Candidate(name=name, x=None, ok=False, conf=0.0, ms=ms0,
                         note="极坐标带越界或整圈无纹理"), np.zeros(N_ANG)
    pp, bp = prep(pp), prep(bp)

    sigma = RADIUS_SIGMA * R
    total = np.zeros(N_ANG, np.float64)
    wsum = 0.0
    fb = np.fft.rfft(bp, axis=-1)          # (nb, C, F)
    fp = np.fft.rfft(pp, axis=-1)          # (np, C, F)
    for i in range(pp.shape[0]):
        # 半径差的权重：贴着边界的一对最重要
        w = np.exp(-((br - pr[i]) / sigma) ** 2)
        keep = w > 0.02
        if not keep.any():
            continue
        cross = fb[keep] * np.conj(fp[i][None, :, :])
        cur = np.fft.irfft(cross, n=N_ANG, axis=-1).sum(axis=1)   # 通道求和
        total += (cur * w[keep][:, None]).sum(axis=0)
        wsum += float(w[keep].sum()) * pp.shape[1]
    ms = (time.perf_counter() - t0) * 1000.0
    if wsum <= 0:
        return Candidate(name=name, x=None, ok=False, conf=0.0, ms=ms,
                         note="没有半径配对落在权重内"), total
    total /= wsum
    pos, sig = _peak(total)
    # 曲线下标 k = 把碎片的角向 profile 循环右移 k 格能对上背景。
    # 下标随角度增大的方向由 `_polar` 里 atan2 的取样决定 —— **不靠推导**，
    # 由 tools/bench_rotate.py 的符号断言（正反两种都算一遍误差）钉死。
    deg = pos * 360.0 / N_ANG
    conf = float(np.clip(sig / PEAK_SCALE, 0.0, 1.0))
    return Candidate(name=name, ok=True, x=deg, conf=conf, ms=ms,
                     note=f"{pp.shape[0]}x{bp.shape[0]} 半径配对，峰显著性 {sig:.2f}"), total


def circ_diff(a: float, b: float) -> float:
    """角度的最短差（带符号，落在 (-180, 180]）。"""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d + 360.0 if d <= -180.0 else d


def _circ_vote(cands: list[Candidate], tolerance: float) -> tuple[float | None, float, int]:
    """角度专用投票 —— **不能用 solver/vote.py**。

    那个投票器按线性距离聚类，而角度是循环量：1° 与 359° 实际只差 2°，
    线性距离却是 358°，两个本来一致的算法会被判成分歧。
    """
    valid = [c for c in cands if c.ok and c.x is not None and c.conf > 0]
    if not valid:
        return None, 0.0, 0
    valid.sort(key=lambda c: c.weight * c.conf, reverse=True)
    best: list[Candidate] = []
    for anchor in valid:
        cl = [c for c in valid if abs(circ_diff(float(c.x), float(anchor.x))) <= tolerance]
        if sum(c.weight * c.conf for c in cl) > sum(c.weight * c.conf for c in best):
            best = cl
    # 循环量求平均要绕参照点做偏移平均，直接算术平均会在 0/360 边界上给出中间的假角度
    ref = float(best[0].x)
    wsum = sum(c.conf for c in best) or 1.0
    off = sum(circ_diff(float(c.x), ref) * c.conf for c in best) / wsum
    x = (ref + off) % 360.0
    conf = max(c.conf for c in best) * (1.0 if len(best) > 1 else 0.72)
    return x, float(np.clip(conf, 0.0, 1.0)), len(best)


def measure_rotation(bg_bgr: np.ndarray, piece_bgra: np.ndarray) -> RotateResult:
    """对外入口：返回"把碎片再顺时针转多少度就对齐"。"""
    t0 = time.perf_counter()
    cands: list[Candidate] = []
    curves: list[np.ndarray] = []
    for grad in (False, True):
        c, curve = algo_polar_corr(bg_bgr, piece_bgra, use_gradient=grad)
        cands.append(c)
        curves.append(curve)
    # 两条判据都是角度，按**循环**容差投票（容差取站点容错 7.2° 的三分之一）
    x, conf, agree = _circ_vote(cands, tolerance=2.4)
    _, _, R = piece_geometry(piece_bgra)
    ms = (time.perf_counter() - t0) * 1000.0
    if x is None:
        return RotateResult(None, 0.0, cands, list(curves[0]), R, ms,
                            "两条判据都没给出角度")
    return RotateResult(float(x) % 360.0, float(conf), cands,
                        [float(v) for v in curves[0]], R, ms,
                        f"投票 {agree}/{len(cands)}")
