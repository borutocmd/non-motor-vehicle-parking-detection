"""检测器：YOLOv8 封装 + 用于自测的 Mock 检测器。

YoloDetector  —— 正式使用，读取 ultralytics 权重
MockDetector  —— 无权重时也能跑通全链路：按颜色找"车"，用于合成视频自测
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    box: tuple[float, float, float, float]   # x1,y1,x2,y2
    cls_id: int
    conf: float
    label: str = ""


class BaseDetector:
    names: dict[int, str] = {}

    def detect(self, frame) -> list[Detection]:
        raise NotImplementedError


class YoloDetector(BaseDetector):
    def __init__(self, weights: str, conf: float = 0.35, iou: float = 0.5,
                 imgsz: int = 640, device: str = "cpu",
                 class_ids: list[int] | None = None,
                 class_names: dict[int, str] | None = None):
        from ultralytics import YOLO  # 延迟导入：mock 模式不需要

        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"权重不存在: {weights}\n"
                f"请把 yolov8n.pt 放到该路径，或用 --mock 跑自测模式。"
            )
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.class_ids = class_ids
        self.names = {int(k): v for k, v in (class_names or {}).items()}
        # 若未指定，用模型自带类别名
        for k, v in getattr(self.model, "names", {}).items():
            self.names.setdefault(int(k), v)

    def detect(self, frame) -> list[Detection]:
        args = dict(conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                    device=self.device, verbose=False)
        if self.class_ids:
            args["classes"] = list(self.class_ids)
        r = self.model.predict(frame, **args)[0]
        out: list[Detection] = []
        if r.boxes is None:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
            out.append(Detection((float(x1), float(y1), float(x2), float(y2)),
                                 int(c), float(p), self.names.get(int(c), str(c))))
        return out


class MockDetector(BaseDetector):
    """颜色阈值检测：合成视频里用纯色矩形代表车辆，无需权重即可验证链路。

    橙色 (BGR 0,140,255)  → class 1 (bicycle)
    蓝色 (BGR 255,120,0)  → class 3 (motorcycle)
    """

    SPEC = [(1, "bicycle", (0, 140, 255)), (3, "motorcycle", (255, 120, 0))]

    def __init__(self, min_area: int = 800):
        self.min_area = min_area
        self.names = {c: n for c, n, _ in self.SPEC}

    def detect(self, frame) -> list[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        out: list[Detection] = []
        for cid, label, bgr in self.SPEC:
            bgr_arr = np.uint8([[bgr]])
            h = int(cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2HSV)[0][0][0])
            lo = np.array([max(h - 12, 0), 90, 90], dtype=np.uint8)
            hi = np.array([min(h + 12, 179), 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                if cv2.contourArea(cnt) < self.min_area:
                    continue
                x, y, w, hh = cv2.boundingRect(cnt)
                out.append(Detection((float(x), float(y), float(x + w), float(y + hh)),
                                     cid, 0.99, label))
        return out


def build_detector(cfg) -> BaseDetector:
    if cfg.model.mock:
        return MockDetector()
    return YoloDetector(
        weights=cfg.path(cfg.model.weights),
        conf=float(cfg.model.conf),
        iou=float(cfg.model.iou),
        imgsz=int(cfg.model.imgsz),
        device=str(cfg.model.device),
        class_ids=list(cfg.model.class_ids) if cfg.model.class_ids else None,
        class_names={int(k): v for k, v in (cfg.model.class_names or {}).items()},
    )
