"""可视化：检测框、禁停区、停留计时、状态面板、告警横幅。

中文渲染说明：OpenCV 自带 putText 不支持中文，这里用 Windows 系统字体 + PIL 画中文，
避免框上出现方块乱码（截图会好看很多）。
"""
from __future__ import annotations

import os

import cv2
import numpy as np

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",   # 黑体
    r"C:\Windows\Fonts\simsun.ttc",
]
_CN_FONT = None


def _cn_font_path() -> str | None:
    global _CN_FONT
    if _CN_FONT is not None:
        return _CN_FONT or None
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            _CN_FONT = p
            return p
    _CN_FONT = ""
    return None


def put_text_cn(img, text: str, org, size: int = 16, color=(255, 255, 255), bold: bool = False):
    """画中文（退化为 cv2.putText 当系统无中文字体）。"""
    path = _cn_font_path()
    if not path:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size / 32.0, color, 1, cv2.LINE_AA)
        return img
    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(path, size)
    xy = (int(org[0]), int(org[1]) - size)
    # 描边提升对比度
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((xy[0] + dx, xy[1] + dy), text, font=font, fill=(0, 0, 0))
    draw.text(xy, text, font=font, fill=(color[2], color[1], color[0]))
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    img[:] = out
    return img


def draw_panel(frame, lines: list[str], origin=(10, 10), size=16, panel_color=(32, 32, 32),
               text_color=(255, 255, 255), alpha=0.55):
    """左上角信息面板。"""
    if not lines:
        return frame
    pad, lh = 8, size + 8
    w = int(max(len(s) for s in lines) * size * 0.62) + pad * 2
    h = lh * len(lines) + pad * 2
    x0, y0 = origin
    roi = frame[y0:y0 + h, x0:x0 + w]
    if roi.size:
        overlay = np.full_like(roi, panel_color)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
        frame[y0:y0 + h, x0:x0 + w] = roi
    for i, s in enumerate(lines):
        put_text_cn(frame, s, (x0 + pad, y0 + pad + (i + 1) * lh - 6), size=size, color=text_color)
    return frame


def draw_detections(frame, tracks, judge_res, names: dict[int, str], colors: dict[str, tuple]):
    """按判定状态给框上色：绿=正常，橙=计时中，红=违停。"""
    for t in tracks:
        status = judge_res.status.get(t.tid, "outside")
        dwell = judge_res.dwell.get(t.tid, 0.0)
        if status == "outside":
            color = colors["ok"]
        elif status in ("inside", "timing"):
            color = colors["warn"]
        elif status == "intrusion":
            color = colors.get("intrusion", colors["bad"])
        else:
            color = colors["bad"]

        x1, y1, x2, y2 = [int(v) for v in t.box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        label = f"#{t.tid} {names.get(t.cls_id, t.cls_id)} {t.conf:.2f}"
        if status in ("inside", "timing"):
            label += f" | 已停 {dwell:.1f}s"
        elif status == "intrusion":
            label += f" | 入侵 {dwell:.1f}s"
        elif status == "violation":
            label += f" | 违停 {dwell:.1f}s"
        label_w = sum(15 if ord(ch) > 127 else 9 for ch in label) + 10
        cy1 = max(y1 - 22, 0)
        cv2.rectangle(frame, (x1, cy1), (x1 + label_w, y1), color, -1)
        put_text_cn(frame, label, (x1 + 4, y1 - 5), size=15, color=(0, 0, 0))

        # 着地点（底边中点）
        ax, ay = int((x1 + x2) / 2), y2
        cv2.circle(frame, (ax, ay), 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (ax, ay), 7, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_alarm_banner(frame, text: str, color=(50, 50, 235), position: str = 'bottom'):
    """告警横幅。默认画在画面底部，避免与目标框上方的标签重叠。"""
    h, w = frame.shape[:2]
    band_h = 46
    y0 = h - band_h if position == 'bottom' else 0
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, y0 + band_h), color, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    put_text_cn(frame, text, (14, y0 + band_h - 12), size=24, color=(255, 255, 255))
    return frame
