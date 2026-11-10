"""用真实 YOLOv8n 权重跑一张真实照片，验证非 mock 路径。"""
import os
import sys
import urllib.request

import cv2
import numpy as np

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)

from src.writer import _imwrite_unicode  # noqa: E402

URL = "https://source.roboflow.com/nmAvWVxki2dFut6a0KOyLtg1sKj2/AHHp1zlxEOoo2MwXXkuX/original.jpg"
OUT = os.path.join(CODE, "samples", "real_test.jpg")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if not os.path.exists(OUT):
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(OUT, "wb") as f:
            f.write(r.read())
        print(f"下载测试图 -> {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")

    from ultralytics import YOLO

    w = os.path.join(CODE, "weights", "yolov8n.pt")
    model = YOLO(w)

    img = cv2.imread(OUT)
    print(f"图片尺寸: {img.shape}")

    r = model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]
    names = model.names
    print(f"检测到 {0 if r.boxes is None else len(r.boxes)} 个目标：")
    canvas = img.copy()
    for b in (r.boxes or []):
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        cid = int(b.cls[0])
        conf = float(b.conf[0])
        name = names.get(cid, str(cid))
        color = (0, 200, 0) if cid in (1, 3) else (200, 200, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        cv2.rectangle(canvas, (x1, max(y1 - 18, 0)), (x1 + len(label) * 9 + 6, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 3, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        print(f"  {name:14} conf={conf:.3f}  box=({x1},{y1},{x2},{y2})")

    out_img = os.path.join(CODE, "outputs", "real_detect.jpg")
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    _imwrite_unicode(out_img, canvas)
    print(f"标注结果 -> {out_img}")


if __name__ == "__main__":
    main()
