"""置信度投票：把多个算法的坐标输出聚类，选出权重最高的簇。

设计要点：
- 每个算法有先验权重 `weight`，实际票重 = weight * conf；
- 同意某个簇的算法越多、簇内票重越高，最终置信度越高；
- 只有一个算法出结果时置信度打折（单点无交叉验证）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    name: str
    x: float | None
    y: float | None = None
    conf: float = 0.0
    ms: float = 0.0
    weight: float = 1.0
    note: str = ""
    ok: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        # 一律 float()：算法里的 conf/x/y 常常是 numpy 标量（OpenCV 出来的
        # np.float32），而 `round(np.float32, 4)` **仍然返回 np.float32** ——
        # 光 round 挡不住。numpy 标量进了事件就会让 `json.dumps` 抛 TypeError，
        # 而那个异常发生在 WS 的 send 里，直接把连接打断（实测 1006）。
        return {
            "name": self.name,
            "ok": bool(self.ok and self.x is not None),
            "x": round(float(self.x), 2) if self.x is not None else None,
            "y": round(float(self.y), 2) if self.y is not None else None,
            "conf": round(float(self.conf), 4),
            "ms": round(float(self.ms), 1),
            "weight": float(self.weight),
            "note": self.note,
            **({"extra": self.extra} if self.extra else {}),
        }


def vote(cands: list[Candidate], tolerance: float = 12.0) -> dict[str, Any]:
    """返回 {x, y, conf, agree, total, picked, clusters}。"""
    valid = [c for c in cands if c.ok and c.x is not None and c.conf > 0]
    total = len(cands)
    if not valid:
        return {"x": None, "y": None, "conf": 0.0, "agree": 0, "total": total,
                "picked": [], "clusters": []}

    # 贪心聚类：按票重降序，逐个吸附到已有簇或自建簇
    valid.sort(key=lambda c: c.weight * c.conf, reverse=True)
    clusters: list[list[Candidate]] = []
    for c in valid:
        for cl in clusters:
            center = sum(m.x for m in cl) / len(cl)
            if abs(c.x - center) <= tolerance:
                cl.append(c)
                break
        else:
            clusters.append([c])

    def cluster_weight(cl: list[Candidate]) -> float:
        return sum(m.weight * m.conf for m in cl)

    clusters.sort(key=cluster_weight, reverse=True)
    best = clusters[0]
    w_sum = sum(m.weight * m.conf for m in best)
    x = sum(m.x * m.weight * m.conf for m in best) / w_sum
    ys = [m.y for m in best if m.y is not None]
    y = sum(m.y * m.weight * m.conf for m in best if m.y is not None) / w_sum if ys else None

    all_weight = sum(c.weight * c.conf for c in valid) or 1.0
    consensus = w_sum / all_weight  # 获胜簇占总票重的比例
    mean_conf = sum(m.conf for m in best) / len(best)
    conf = mean_conf * (0.55 + 0.45 * consensus)
    if len(best) == 1:
        conf *= 0.72  # 孤证打折
    if len(clusters) > 1:
        # 存在明显分歧时再打一点折
        conf *= 0.85 + 0.15 * consensus

    return {
        "x": round(float(x), 2),
        "y": round(float(y), 2) if y is not None else None,
        "conf": round(float(min(1.0, conf)), 4),
        "agree": len(best),
        "total": total,
        "picked": [m.name for m in best],
        "clusters": [
            {"center": round(float(sum(m.x for m in cl) / len(cl)), 2),
             "weight": round(float(cluster_weight(cl)), 4),
             "members": [m.name for m in cl]}
            for cl in clusters
        ],
    }


def vote2d(cands: list[Candidate], tolerance: float = 18.0) -> dict[str, Any]:
    """点选题型用：按 (x, y) 欧氏距离聚类。"""
    import math

    valid = [c for c in cands if c.ok and c.x is not None and c.y is not None and c.conf > 0]
    total = len(cands)
    if not valid:
        return {"x": None, "y": None, "conf": 0.0, "agree": 0, "total": total, "picked": []}
    valid.sort(key=lambda c: c.weight * c.conf, reverse=True)
    clusters: list[list[Candidate]] = []
    for c in valid:
        for cl in clusters:
            cx = sum(m.x for m in cl) / len(cl)
            cy = sum(m.y for m in cl) / len(cl)
            if math.hypot(c.x - cx, c.y - cy) <= tolerance:
                cl.append(c)
                break
        else:
            clusters.append([c])
    clusters.sort(key=lambda cl: sum(m.weight * m.conf for m in cl), reverse=True)
    best = clusters[0]
    w_sum = sum(m.weight * m.conf for m in best) or 1.0
    all_w = sum(c.weight * c.conf for c in valid) or 1.0
    mean_conf = sum(m.conf for m in best) / len(best)
    conf = mean_conf * (0.55 + 0.45 * w_sum / all_w)
    if len(best) == 1:
        conf *= 0.72
    return {
        "x": round(sum(m.x * m.weight * m.conf for m in best) / w_sum, 2),
        "y": round(sum(m.y * m.weight * m.conf for m in best) / w_sum, 2),
        "conf": round(min(1.0, conf), 4),
        "agree": len(best),
        "total": total,
        "picked": [m.name for m in best],
    }
