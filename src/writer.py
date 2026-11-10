"""取证落盘：违停截图 + CSV 记录。

截图命名: violation_20261030_201533_t12_lot_A.jpg
  时间戳 / 轨迹 ID / 区域名 —— 便于事后核查
CSV 字段: 时间, 视频时间轴(秒), 区域, 轨迹ID, 类别, 置信度, 停留时长, 截图文件
"""
from __future__ import annotations

import csv
import os
from datetime import datetime

import cv2


def _imwrite_unicode(path: str, img, quality: int = 92) -> bool:
    """cv2.imwrite 在 Windows 上写不了中文/非 ASCII 路径，改用 imencode + 文件写入。"""
    ext = os.path.splitext(path)[1] or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


class ViolationWriter:
    FIELDS = ["时间", "视频时间轴s", "事件类型", "区域", "轨迹ID", "类别", "置信度", "停留时长s", "着地点", "截图文件"]

    def __init__(self, out_dir: str, prefix: str = "violation",
                 save_csv: bool = True, csv_name: str = "violations.csv",
                 save_screenshot: bool = True):
        self.out_dir = out_dir
        self.shot_dir = os.path.join(out_dir, "screenshots")
        self.prefix = prefix
        self.save_csv = save_csv
        self.save_screenshot = save_screenshot
        self.csv_path = os.path.join(out_dir, csv_name)
        os.makedirs(self.shot_dir, exist_ok=True)
        if save_csv:
            new = not os.path.exists(self.csv_path)
            self._fh = open(self.csv_path, "a", newline="", encoding="utf-8-sig")
            self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
            if new:
                self._writer.writeheader()
        else:
            self._fh, self._writer = None, None

    def save(self, frame, event, cls_name: str, video_ts: float, kind_cn: str = "") -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{self.prefix}_{stamp}_t{event.tid}_{event.zone}.jpg"
        path = ""
        if self.save_screenshot:
            path = os.path.join(self.shot_dir, fname)
            _imwrite_unicode(path, frame)

        if self._writer:
            self._writer.writerow({
                "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "视频时间轴s": f"{video_ts:.2f}",
                "事件类型": kind_cn,
                "区域": event.zone,
                "轨迹ID": event.tid,
                "类别": cls_name,
                "置信度": f"{event.conf:.3f}",
                "停留时长s": f"{event.dwell:.2f}",
                "着地点": f"({event.anchor[0]:.0f},{event.anchor[1]:.0f})",
                "截图文件": os.path.basename(path) if path else "",
            })
            self._fh.flush()
        return path

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
