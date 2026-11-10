"""配置加载：把 YAML 读成带默认值的对象，避免代码里散落魔法数字。"""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "input": {"source": "samples/parking_demo.mp4", "loop": False},
    "model": {
        "weights": "weights/yolov8n.pt", "conf": 0.35, "iou": 0.5, "imgsz": 640,
        "device": "cpu", "class_ids": [], "class_names": {},
        "mock": False,
    },
    "roi": {"polygons_file": "configs/roi.json", "active": "lot_A", "draw_hint": True},
    "judge": {
        "anchor": "bottom_center", "dwell_seconds": 8.0, "min_hits": 5,
        "cooldown_seconds": 30.0, "require_inside": True,
    },
    "tracker": {"iou_threshold": 0.3, "max_age": 30},
    "output": {
        "dir": "outputs", "save_screenshot": True, "screenshot_prefix": "violation",
        "save_csv": True, "csv_name": "violations.csv", "save_annotated_video": False,
        "show_window": True, "window_scale": 1.0,
    },
    "visual": {
        "font_scale": 0.55, "thickness": 2,
        "color_roi": [0, 215, 255], "color_roi_fill": [0, 215, 255],
        "color_ok": [80, 220, 100], "color_warn": [0, 165, 255], "color_bad": [50, 50, 235],
        "color_text": [255, 255, 255], "color_panel": [32, 32, 32],
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """点号访问的配置对象：cfg.judge.dwell_seconds"""

    def __init__(self, data: dict, root: str = "."):
        self._data = data
        self.root = root

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        v = self._data.get(item)
        return Config(v, self.root) if isinstance(v, dict) else v

    def get(self, key, default=None):
        return self._data.get(key, default)

    # 让配置对象也能当 dict 用：嵌套 dict 会被包装成 Config，
    # 如果不提供下面这些方法，外层 .items() / len() 之类会失效。
    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def __getitem__(self, k):
        return self._data[k]

    def __contains__(self, k):
        return k in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"Config({self._data})"

    def as_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def path(self, p: str) -> str:
        """把配置里的相对路径解析成基于项目根目录的绝对路径。"""
        if not p:
            return p
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(self.root, p))


def load_config(path: str | None = None) -> Config:
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    root = os.path.dirname(os.path.abspath(path))
    data = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    return Config(_merge(DEFAULTS, data), root=root)
