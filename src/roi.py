"""禁停区多边形：加载、缩放、命中判定、绘制。

坐标系约定
----------
配置文件里的点可以是：
  * 归一化坐标（0~1 的小数）—— 适应任意分辨率，推荐
  * 绝对像素（整数）—— 适配固定机位
加载时自动判断：所有坐标 <= 1.0 视为归一化。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np


@dataclass
class RoiZone:
    name: str
    points_norm: list[tuple[float, float]] = field(default_factory=list)
    normalized: bool = True

    def points(self, w: int, h: int) -> list[tuple[int, int]]:
        if self.normalized:
            return [(int(round(x * w)), int(round(y * h))) for x, y in self.points_norm]
        return [(int(round(x)), int(round(y))) for x, y in self.points_norm]

    def contains(self, pt: tuple[float, float], w: int, h: int, eps: float = 0.0) -> bool:
        """点是否在多边形内（含边界）。eps>0 时允许外扩容差（像素）。"""
        poly = np.array(self.points(w, h), dtype=np.int32)
        if len(poly) < 3:
            return False
        d = cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(pt[0]), float(pt[1])), True)
        return d >= -abs(eps)

    def area(self, w: int, h: int) -> float:
        poly = np.array(self.points(w, h), dtype=np.float32)
        return float(cv2.contourArea(poly))


def load_zones(path: str) -> dict[str, RoiZone]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    zones: dict[str, RoiZone] = {}
    # 兼容两种写法：{"lot_A": [[x,y],...]}  或  {"zones":[{"name":..,"points":..}]}
    items = raw.get("zones") if isinstance(raw, dict) and "zones" in raw else raw
    if isinstance(items, dict):
        for name, pts in items.items():
            zones[name] = _build(name, pts)
    elif isinstance(items, list):
        for it in items:
            zones[it["name"]] = _build(it["name"], it["points"])
    return zones


def _build(name: str, pts: Sequence) -> RoiZone:
    if pts and isinstance(pts[0], dict):
        pts = [(p["x"], p["y"]) for p in pts]
    pts = [(float(p[0]), float(p[1])) for p in pts]
    normalized = all(0.0 <= v <= 1.0 for p in pts for v in p)
    return RoiZone(name=name, points_norm=pts, normalized=normalized)


def draw_zone(frame, zone: RoiZone, color=(0, 215, 255), alpha: float = 0.18,
              thickness: int = 2, label: str | None = None, font_scale: float = 0.55):
    """画半透明填充 + 实线边界 + 名称标签。"""
    h, w = frame.shape[:2]
    poly = np.array(zone.points(w, h), dtype=np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [poly], True, color, thickness, cv2.LINE_AA)

    text = label if label is not None else zone.name
    if text:
        x, y = poly[0]
        y = max(y - 8, 16)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    color, thickness, cv2.LINE_AA)

    # 每个顶点画小方块，方便调点
    for p in poly:
        cv2.circle(frame, tuple(int(v) for v in p), 4, color, -1, cv2.LINE_AA)
    return frame


def anchor_of_box(box: tuple[float, float, float, float], mode: str = "bottom_center") -> tuple[float, float]:
    """检测框 → 车辆"着地点"。box = (x1, y1, x2, y2)。"""
    x1, y1, x2, y2 = box
    if mode == "center":
        return (x1 + x2) / 2, (y1 + y2) / 2
    if mode == "bottom_left":
        return x1, y2
    if mode == "bottom_right":
        return x2, y2
    return (x1 + x2) / 2, y2  # bottom_center：默认，最贴近车轮触地位置
