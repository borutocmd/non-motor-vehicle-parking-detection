"""用数据集里的一张真实照片，生成"固定机位监控"演示视频。

为什么这样做
------------
直接把车"贴"到别的背景上会有硬边和尺度不协调的问题。更干净的做法是：
取一张**本来就有电动车停放**的真实照片，做缓慢推移（Ken Burns）+ 轻微噪点，
模拟固定机位的监控画面。车辆由真实照片提供，YOLO 能正常检出，
"进入即告警 → 停留超时升级"的判定也完全真实。

用法
----
python tools/build_demo_video.py --out samples/scene_demo.mp4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import cv2
import numpy as np

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(CODE, "..", "data", "ebike_scene")
W, H, FPS = 1280, 720, 25


def imread_unicode(path: str):
    # cv2.imread 在 Windows 上读不了中文路径，用 imdecode 绕过
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def load_items(split: str = "train"):
    items = []
    for f in glob.glob(os.path.join(DATA, split, "images", "*.jpg")):
        lbl = os.path.join(DATA, split, "labels",
                           os.path.splitext(os.path.basename(f))[0] + ".txt")
        if not os.path.exists(lbl):
            continue
        rows = []
        for line in open(lbl, encoding="utf-8"):
            p = line.split()
            if len(p) >= 5:
                rows.append(tuple(float(v) for v in p[1:5]))
        if rows:
            items.append((f, rows))
    return items


def center_crop_169(img):
    """居中裁成 16:9，避免出现黑边。"""
    h, w = img.shape[:2]
    target = W / H
    if w / h > target:
        nw = int(h * target)
        x0 = (w - nw) // 2
        return img[:, x0:x0 + nw]
    nh = int(w / target)
    y0 = max(0, (h - nh) // 2)
    return img[y0:y0 + nh, :]


def pick_photo(items):
    """挑一张车居中、占比合适的照片。"""
    best = None
    for path, rows in items:
        big = max(rows, key=lambda r: r[2] * r[3])
        area = big[2] * big[3]
        if not (0.035 <= area <= 0.115):
            continue
        if not (0.28 <= big[0] <= 0.75):
            continue
        score = -abs(area - 0.08) + min(len(rows), 5) * 0.02 - abs(big[0] - 0.5) * 0.05
        if best is None or score > best[0]:
            best = (score, path, big)
    if best is None:
        raise SystemExit("没有合适的照片（检查数据集路径）")
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(CODE, "samples", "scene_demo.mp4"))
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--zoom", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    items = load_items("train")
    path, box = pick_photo(items)

    raw = imread_unicode(path)
    if raw is None:
        raise SystemExit(f"读图失败: {path}")
    base = center_crop_169(raw)
    base = cv2.resize(base, (W, H), interpolation=cv2.INTER_AREA)
    bh, bw = base.shape[:2]
    rawh, raww = raw.shape[:2]

    # 车在原图的归一化框 → 裁切并缩放后的画面坐标
    if raww / rawh > W / H:
        off_x = (raww - int(rawh * W / H)) // 2
        off_y = 0
        scale = bh / rawh
    else:
        off_x = 0
        off_y = (rawh - int(raww * H / W)) // 2
        scale = bw / raww

    def to_frame(nx, ny):
        return (nx * raww - off_x) * scale, (ny * rawh - off_y) * scale

    ox1 = box[0] - box[2] / 2
    oy1 = box[1] - box[3] / 2
    fx1, fy1 = to_frame(ox1, oy1)
    fx2, fy2 = to_frame(ox1 + box[2], oy1 + box[3])

    pad_x = (fx2 - fx1) * 0.45
    pad_y = (fy2 - fy1) * 0.25
    zx1 = max(0.05, min(0.45, (fx1 - pad_x) / W))
    zx2 = min(0.95, max(0.55, (fx2 + pad_x) / W))
    zy1 = max(0.08, min(0.50, (fy1 - pad_y) / H))
    zy2 = min(0.94, max(0.60, (fy2 + pad_y * 1.6) / H))
    zone = [[round(zx1, 3), round(zy1, 3)], [round(zx2, 3), round(zy1, 3)],
            [round(zx2, 3), round(zy2, 3)], [round(zx1, 3), round(zy2, 3)]]

    total = int(args.seconds * FPS)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rng = np.random.default_rng(args.seed)

    for i in range(total):
        p = i / max(1, total - 1)
        z = 1.0 + (args.zoom - 1.0) * p
        cw, ch = int(bw / z), int(bh / z)
        cx = (bw - cw) // 2
        cy = (bh - ch) // 2
        crop = base[cy:cy + ch, cx:cx + cw]
        frame = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)

        noise = rng.normal(0, 1.6, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        k = 1.0 + 0.012 * np.sin(2 * np.pi * i / 120.0)
        frame = np.clip(frame.astype(np.float32) * k, 0, 255).astype(np.uint8)

        cv2.putText(frame, "CAM-01  2026-11-06 15:20:11", (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA)
        vw.write(frame)

    vw.release()

    meta = {
        "video": os.path.abspath(args.out),
        "source_photo": os.path.basename(path),
        "roi": zone,
        "seconds": args.seconds,
        "fps": FPS,
        "frames": total,
        "note": "固定机位模拟视频，车辆来自真实照片，可直接用 COCO 预训练模型检测",
    }
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"已生成 {args.out}")
    print(f"  源照片: {os.path.basename(path)}")
    print(f"  建议禁停区: {zone}")
    print(f"  帧数: {total}  时长: {args.seconds}s")


if __name__ == "__main__":
    main()
