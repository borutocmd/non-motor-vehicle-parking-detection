"""轻量 IoU 跟踪器：给检测框分配稳定 ID，用于停留时长累计。

不依赖 deep-sort / byte-track 这类重依赖，纯 NumPy 实现，教程和报告里好讲。
匹配策略：贪心 IoU（先算矩阵，再按分数从高到低配对）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) b: (M,4) 均为 x1y1x2y2，返回 (N,M) IoU。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)

    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih

    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter + 1e-9
    return inter / union


@dataclass
class Track:
    tid: int
    box: tuple[float, float, float, float]
    cls_id: int
    conf: float
    hits: int = 1
    age: int = 0
    missed: int = 0
    first_ts: float = 0.0
    last_ts: float = 0.0
    # 供 judge 使用的附加状态
    extra: dict = field(default_factory=dict)


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, dets: list[dict], ts: float = 0.0) -> list[Track]:
        """dets: [{'box':(x1,y1,x2,y2), 'cls_id':int, 'conf':float}, ...]"""
        for t in self.tracks:
            t.age += 1

        if not dets:
            for t in self.tracks:
                t.missed += 1
            self.tracks = [t for t in self.tracks if t.missed <= self.max_age]
            return [t for t in self.tracks if t.missed == 0]

        det_boxes = np.array([d["box"] for d in dets], dtype=np.float32)
        trk_boxes = np.array([t.box for t in self.tracks], dtype=np.float32) if self.tracks else np.zeros((0, 4), np.float32)
        ious = iou_matrix(trk_boxes, det_boxes)

        matched_t: set[int] = set()
        matched_d: set[int] = set()

        if ious.size:
            order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
            for ti, di in order:
                if ious[ti, di] < self.iou_threshold:
                    continue
                if ti in matched_t or di in matched_d:
                    continue
                matched_t.add(int(ti))
                matched_d.add(int(di))
                t = self.tracks[int(ti)]
                t.box = tuple(float(v) for v in det_boxes[int(di)])
                t.cls_id = int(dets[int(di)]["cls_id"])
                t.conf = float(dets[int(di)]["conf"])
                t.hits += 1
                t.missed = 0
                t.last_ts = ts

        # 未匹配的轨迹
        for ti, t in enumerate(self.tracks):
            if ti not in matched_t:
                t.missed += 1

        # 未匹配的检测 → 新轨迹
        for di, d in enumerate(dets):
            if di in matched_d:
                continue
            self.tracks.append(Track(
                tid=self._next_id,
                box=tuple(float(v) for v in d["box"]),
                cls_id=int(d["cls_id"]),
                conf=float(d["conf"]),
                first_ts=ts,
                last_ts=ts,
            ))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]
        return [t for t in self.tracks if t.missed == 0]
