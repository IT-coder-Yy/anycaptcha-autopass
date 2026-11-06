"""极验第四代「滑动拼图」：在底图上找那块碎片该去的位置。

## 为什么不复用 `solver/detect.py`

那一套是照 tianai 的题面写的：缺口是**在底图上挖掉并压暗**的洞，所以
"残留纹理""填充色""列投影"这些判据成立，公共裁判也是按"洞"标定的。
极验第四代的题面结构不一样（实测，见 `probe_out/geetest.md`）：

- 下发的 `bg/<hash>.png` 是**没有洞的完整图**（把碎片内容拿去和它做 SQDIFF
  全图搜索，最小 RMSE 66、中位 85 —— 根本不存在"挖掉"这回事）；
- 缺口是在原图上叠了一层**淡淡的三角形色罩**，内容还看得见；
- 碎片 `slice/<hash>.png` 是**未加色罩的原始内容**，带 alpha 的三角形。

于是判据只有一条是硬的：**碎片内容与底图在缺口处逐像素对应**，只差一个
接近仿射的亮度变化（色罩）。对亮度不敏感的相关就是 `TM_CCOEFF_NORMED` + mask。
实测一道真题：峰值 0.581 @ x=211，次峰 0.357 —— 1.6 倍差距，很干净。
（同一道题用 `TM_SQDIFF` 的最小值落在 x=154，是错的；用 `TM_CCORR_NORMED`
最大值落在 x=134，也是错的。所以方法选择不是风格问题。）

## 一条免费的强否决

碎片只能**水平**移动，所以匹配出来的 y 必须等于碎片当前的 y。
实测那道题：碎片当前 y=127，匹配到 y=126（差 1px，来自 bbox 取整）。
这条比任何置信度数字都硬 —— y 对不上就是匹配错了，直接弃权换题。
所以搜索本身只在 `cur_y ± Y_BAND` 这几行里做，等于把二维搜索压成一维。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from common.vote import Candidate

# 只在碎片当前 y 的上下这么多行里搜。取 6 是给 bbox 取整与 DPR 缩放留的余量，
# 不是给"碎片可能上下动"留的 —— 它不会。
Y_BAND = 6
# 匹配到的 y 与当前 y 差超过这个就判匹配失败（见模块 docstring 的"免费强否决"）
Y_TOL = 3.0

# 置信度标定：见 tools/bench_geetest.py。峰值与次峰的比值是主要判据 ——
# 极验的底图是插画风、大块同色区域多，绝对峰值本身不稳，比值才稳。
MARGIN_SCALE = 0.28     # (峰 - 次峰) / 峰 除以它，压到 0-1
PEAK_FLOOR = 0.30       # 峰值低于这个直接不信（实测正确匹配 0.58，错的都在 0.36 以下）
EXCLUDE = 24            # 找次峰时在主峰左右各排除多少列
AGREE_PX = 6.0          # 两条判据的 x 差在这个范围内算"落在同一处"

# 色罩判据（`_match_tint`）的两个常量。实测 39 道离线样本：内容相关在灰底、
# 大块同色的题面上会失手（39 道里 16 道弃权，其中 11 道是两条内容判据分歧），
# 而**恰恰在那些题面上色罩看得最清楚** —— 缺口是一块明显压暗的形状。
# 所以这一条不是冗余，是补盲区：它只看"这块形状比 bbox 里其余地方暗多少"，
# 压根不看碎片的内容，与两条内容判据完全独立。
TINT_FLOOR = 0.045      # 暗度对比低于这个不算见到色罩
TINT_SCALE = 0.30       # 余量归一化尺度

GEETEST_GATE = 0.45     # 换题闸门，标定见 tools/bench_geetest.py


@dataclass
class SliceResult:
    """`x` 是碎片**应当移到**的位置（底图坐标，与 `cur_x` 同口径）。"""

    x: float | None
    y: float | None = None
    conf: float = 0.0
    move: float | None = None          # 需要的水平位移 = x - cur_x
    candidates: list[Candidate] = field(default_factory=list)
    ms: float = 0.0
    note: str = ""
    debug: dict[str, Any] = field(default_factory=dict)


def slice_template(slice_bgra: np.ndarray, alpha_min: int = 200
                   ) -> tuple[np.ndarray, np.ndarray, int, int]:
    """从碎片 PNG 里取出"完全不透明的那块"，返回 (模板, 掩膜, 模板在碎片图里的 x, y)。

    只用 alpha 高的核心区：边缘那一圈半透明像素是抗锯齿与描边，内容已经被混过，
    拿它去比会把真峰压下去。实测一道真题 80x80 的碎片里 alpha==255 有 1845 个像素。
    """
    a = slice_bgra[:, :, 3]
    m = (a >= alpha_min).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) < 60:
        raise ValueError(f"碎片不透明像素太少（{len(xs)}），拿不到模板")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (slice_bgra[y0:y1, x0:x1, :3].copy(),
            m[y0:y1, x0:x1].copy(), x0, y0)


def _grad(img: np.ndarray) -> np.ndarray:
    """梯度幅值（单通道 float32）。与原始 BGR 互为独立判据：

    色罩改变的是亮度，边缘结构不受影响；反过来，大面积同色区域上 BGR 相关很强
    但梯度相关几乎为 0 —— 两条一起投票，能把"色块碰巧对上"筛掉。
    """
    g = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _peak_of(col: np.ndarray, floor: float, scale: float, name: str,
             ms: float, what: str) -> tuple[Candidate, int]:
    """从逐列最优曲线里取峰、次峰，算置信度。三条判据共用这一段。"""
    k = int(np.argmax(col))
    peak = float(col[k])
    if peak < floor:
        return (Candidate(name=name, x=float(k), ok=False, conf=0.0, ms=ms,
                          note=f"{what} {peak:.3f} < 下限 {floor}"), k)
    lo, hi = max(0, k - EXCLUDE), min(len(col), k + EXCLUDE + 1)
    rest = np.concatenate([col[:lo], col[hi:]])
    runner = float(rest.max()) if len(rest) else -1.0
    margin = (peak - runner) / max(abs(peak), 1e-6)
    conf = float(np.clip(margin / scale, 0.0, 1.0))
    return (Candidate(name=name, x=float(k), ok=True, conf=conf, ms=ms,
                      note=f"{what} {peak:.3f}@x={k} 次峰 {runner:.3f} 余量 {margin:.3f}"), k)


def _match_band(field_: np.ndarray, tpl: np.ndarray, msk: np.ndarray,
                cur_y: int, name: str) -> tuple[Candidate, np.ndarray, float]:
    """在 `cur_y ± Y_BAND` 这条横带里做带掩膜的 CCOEFF 相关。

    返回 (候选, 逐列最优值曲线, 最优行的绝对 y)。
    """
    t0 = time.perf_counter()
    h = tpl.shape[0]
    top = max(0, cur_y - Y_BAND)
    bot = min(field_.shape[0], cur_y + Y_BAND + h)
    band = field_[top:bot]
    if band.shape[0] < h + 1:
        return (Candidate(name=name, x=None, ok=False, conf=0.0,
                          note="碎片贴边，横带高度不够"), np.zeros(0), float(cur_y))
    r = cv2.matchTemplate(band, tpl, cv2.TM_CCOEFF_NORMED, mask=msk)
    r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
    col = r.max(axis=0)                       # 每一列在带内的最优
    row = r.argmax(axis=0)
    ms = (time.perf_counter() - t0) * 1000.0
    cand, k = _peak_of(col, PEAK_FLOOR, MARGIN_SCALE, name, ms, "峰")
    return cand, col, float(top + int(row[k]))


def _match_tint(gray: np.ndarray, msk: np.ndarray, cur_y: int, name: str
                ) -> tuple[Candidate, np.ndarray, float]:
    """色罩判据：找"这块碎片形状比它的 bbox 里其余地方暗多少"最大的那一列。

    **完全不看碎片的内容**，所以与两条内容判据独立。用 bbox 内的非形状区域当参照，
    不用向外扩环 —— 拼图块的 bbox 里本来就有大片非形状区域（实测 80x80 的碎片里
    alpha==255 只有 1845 个像素），既省一次卷积，也让模板尺寸与内容判据完全一致，
    横带的行号换算可以照抄。

    `TM_CCORR` 在模板是 0/1 掩膜时算的就是"掩膜下的像素和"，于是两次相关就能拿到
    形状内均值与形状外均值。
    """
    t0 = time.perf_counter()
    h = tpl_h = msk.shape[0]
    top = max(0, cur_y - Y_BAND)
    bot = min(gray.shape[0], cur_y + Y_BAND + tpl_h)
    band = gray[top:bot]
    if band.shape[0] < h + 1:
        return (Candidate(name=name, x=None, ok=False, conf=0.0,
                          note="碎片贴边，横带高度不够"), np.zeros(0), float(cur_y))
    inv = 1.0 - msk
    a_in, a_out = float(msk.sum()), float(inv.sum())
    if a_in < 40 or a_out < 40:
        return (Candidate(name=name, x=None, ok=False, conf=0.0,
                          note="掩膜内外像素太少，色罩判据不成立"), np.zeros(0), float(cur_y))
    s_in = cv2.matchTemplate(band, msk, cv2.TM_CCORR)
    s_out = cv2.matchTemplate(band, inv, cv2.TM_CCORR)
    m_in, m_out = s_in / a_in, s_out / a_out
    r = (m_out - m_in) / (m_out + 8.0)        # 形状内更暗 → 正
    r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
    col = r.max(axis=0)
    row = r.argmax(axis=0)
    ms = (time.perf_counter() - t0) * 1000.0
    cand, k = _peak_of(col, TINT_FLOOR, TINT_SCALE, name, ms, "暗度对比")
    return cand, col, float(top + int(row[k]))



def locate_slice(bg_bgr: np.ndarray, slice_bgra: np.ndarray,
                 cur_x: float, cur_y: float) -> SliceResult:
    """求碎片该移到哪儿。`cur_x/cur_y` 是碎片图**左上角**在底图坐标里的当前位置。

    三条判据投票，要求至少两条落在同一处（±`AGREE_PX`）：

      · `ccoeff_bgr`  —— 碎片内容与底图的带掩膜 CCOEFF（对亮度仿射变化不敏感）；
      · `ccoeff_grad` —— 同上但比的是梯度幅值（色罩不改边缘结构）；
      · `tint`        —— 碎片形状比其 bbox 其余部分暗多少（只看底图，不看碎片内容）。

    前两条在纹理丰富的题面上很硬，第三条专治灰底、大块同色的题面 —— 实测那类题面
    上前两条会互相分歧，而色罩恰好在那里最明显（见 `TINT_FLOOR` 的注释）。

    返回的 `move` 就是把手要走的位移（这一档碎片与把手是 1:1，
    几何上 `(窗口宽 - 碎片宽) / (轨道宽 - 把手宽) = (300-80)/(300-80) = 1`，
    场景层仍会实测一次并记日志）。
    """
    t0 = time.perf_counter()
    tpl, msk, tx, ty = slice_template(slice_bgra)
    # 模板原点在底图里的当前位置
    ox, oy = cur_x + tx, cur_y + ty
    msk_f = msk.astype(np.float32)
    gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    cands: list[Candidate] = []
    curves: dict[str, list[float]] = {}
    ys: dict[str, float] = {}
    runs: list[tuple[str, Any]] = [
        ("ccoeff_bgr", (bg_bgr.astype(np.float32), tpl.astype(np.float32))),
        ("ccoeff_grad", (_grad(bg_bgr), _grad(tpl))),
        ("tint", None),
    ]
    for name, args in runs:
        if name == "tint":
            c, col, by = _match_tint(gray, msk_f, int(round(oy)), name)
        else:
            c, col, by = _match_band(args[0], args[1], msk_f, int(round(oy)), name)
        cands.append(c)
        curves[name] = [float(v) for v in col]
        if c.ok:
            ys[name] = by

    ok = [c for c in cands if c.ok and c.x is not None]
    ms = (time.perf_counter() - t0) * 1000.0
    if not ok:
        return SliceResult(None, None, 0.0, None, cands, ms,
                           "三条判据都没给出可信峰", {"curves": curves})

    # 按位置聚类：取最大的一簇，并列时取总置信度高的那簇
    def cluster(seed: Candidate) -> list[Candidate]:
        return [c for c in ok if abs(float(c.x) - float(seed.x)) <= AGREE_PX]

    best = max((cluster(c) for c in ok),
               key=lambda g: (len(g), sum(x.conf for x in g)))
    if len(best) < 2:
        spread = " / ".join(f"{c.name}@{c.x:.0f}" for c in ok)
        return SliceResult(None, None, 0.0, None, cands, ms,
                           f"{len(ok)} 条判据没有两条落在同一处（{spread}）",
                           {"curves": curves})

    xs = sorted(float(c.x) for c in best)
    target = float(np.median(xs))
    # 三条全同意时额外加一点信心；两条同意时按均值 + 固定加成（与旧版同一口径）
    conf = float(min(1.0, sum(c.conf for c in best) / len(best)
                     + (0.10 if len(best) == 2 else 0.20)))

    # y 一致性否决取**簇内偏差最小的那条**判据，不是置信度最高的那条。
    # 实测 39 道（`tools/bench_geetest.py`）：`ccoeff_bgr` 的 |dy| 中位 1.4、p90 4.7、
    # 最大 6.4，而 `ccoeff_grad` 是 0.6 / 1.6 / 2.6 —— BGR 相关在竖直方向平滑的题面上
    # 峰会在横带里上下漂。取置信度最高的那条会把 4 道**位置本来就对的**题误否
    # （人眼验收过：碎片正好落在缺口里）。取簇内最小值后，两条内容判据一致的
    # 27 道里 min|dy| 中位 0.6、p90 1.3、最大 1.6 —— 3.0px 这条线因此仍然有意义。
    per = [(abs(ys[c.name] - oy), c) for c in best if c.name in ys]
    if per:
        dy, keeper = min(per, key=lambda p: p[0])
        y_hit = ys[keeper.name]
    else:
        dy, y_hit = 0.0, oy
    if dy > Y_TOL:
        # 免费的强否决：碎片只能水平动，y 对不上就是匹配错了
        return SliceResult(None, float(y_hit), 0.0, None, cands, ms,
                           f"匹配到的 y={y_hit:.0f} 与碎片当前 y={oy:.0f} 差 {dy:.1f}px，"
                           f"超过 {Y_TOL}px —— 判为误匹配", {"curves": curves})

    names = "+".join(c.name for c in best)
    return SliceResult(target, float(y_hit), float(conf), target - ox, cands, ms,
                       f"y 一致（差 {dy:.1f}px），{len(best)}/3 条判据同意（{names}）",
                       {"curves": curves, "origin": [ox, oy], "tpl": list(tpl.shape[:2])})
