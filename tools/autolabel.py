"""用 YOLOv8 预训练模型对抽出的帧做「自动预标注」，生成 YOLO 格式标签。

思路：预训练模型已经能检测 bicycle / motorcycle，直接把它在**你的场景**上的预测
当作初始标注，人工只需要抽查剔除误检、补漏检 —— 比从零画框快 10 倍以上。

用法
----
python tools/autolabel.py --frames datasets/raw_frames --out datasets/labeled --conf 0.35
python tools/autolabel.py --frames datasets/raw_frames --out datasets/labeled --classes 1,3,0

类别映射（输出统一为）：0=bicycle 1=motorcycle 2=person
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.writer import _imwrite_unicode  # noqa: E402

# COCO id -> 输出 id
MAP = {1: 0, 3: 1, 0: 2}
NAMES = ["bicycle", "motorcycle", "person"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="抽帧目录")
    ap.add_argument("--out", required=True, help="输出目录（images/ labels/ preview.jpg）")
    ap.add_argument("--weights", default="weights/yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--classes", default="1,3", help="要保留的 COCO 类别，逗号分隔")
    ap.add_argument("--min-area", type=float, default=0.0015, help="框面积占比下限，过滤极小误检")
    ap.add_argument("--preview", type=int, default=12, help="生成预览图的目标数（0=不生成）")
    args = ap.parse_args()

    keep = {int(x) for x in args.classes.split(",") if x.strip()}
    img_dir = os.path.join(args.out, "images")
    lbl_dir = os.path.join(args.out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(args.weights)

    files = sorted(f for f in os.listdir(args.frames) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        raise SystemExit(f"{args.frames} 里没有图片")

    stats = {"images": 0, "with_label": 0, "boxes": 0, "per_class": {n: 0 for n in NAMES}}
    preview_items = []

    for i, fn in enumerate(files, 1):
        src = os.path.join(args.frames, fn)
        img = cv2.imread(src)
        if img is None:
            continue
        h, w = img.shape[:2]
        r = model.predict(img, imgsz=640, conf=args.conf, verbose=False)[0]

        lines, drawn = [], []
        for b in (r.boxes or []):
            cid = int(b.cls[0])
            if cid not in keep or cid not in MAP:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            if ((x2 - x1) * (y2 - y1)) / (w * h) < args.min_area:
                continue
            out_id = MAP[cid]
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{out_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            drawn.append((int(x1), int(y1), int(x2), int(y2), NAMES[out_id]))
            stats["per_class"][NAMES[out_id]] += 1

        stem = os.path.splitext(fn)[0]
        _imwrite_unicode(os.path.join(img_dir, fn), img)
        with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        stats["images"] += 1
        stats["boxes"] += len(lines)
        if lines:
            stats["with_label"] += 1
            if len(preview_items) < args.preview:
                vis = img.copy()
                for x1, y1, x2, y2, nm in drawn:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(vis, nm, (x1 + 3, max(y1 - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                preview_items.append(vis)

        if i % 50 == 0:
            print(f"  {i}/{len(files)}  已标注 {stats['with_label']} 张 / {stats['boxes']} 框")

    # 预览拼图
    if preview_items:
        th = 260
        tiles = []
        for v in preview_items:
            hh, ww = v.shape[:2]
            scale = th / hh
            tiles.append(cv2.resize(v, (int(ww * scale), th)))
        cols = 3
        rows = (len(tiles) + cols - 1) // cols
        cw = max(t.shape[1] for t in tiles)
        sheet = 255 * __import__("numpy").ones((rows * th, cols * cw, 3), dtype="uint8")
        for idx, t in enumerate(tiles):
            r_, c_ = divmod(idx, cols)
            sheet[r_ * th:r_ * th + th, c_ * cw:c_ * cw + t.shape[1]] = t
        _imwrite_unicode(os.path.join(args.out, "preview.jpg"), sheet)
        print(f"  预览图 -> {os.path.join(args.out, 'preview.jpg')}")

    with open(os.path.join(args.out, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(NAMES) + "\n")

    print("\n[done]")
    print(f"  处理图片 {stats['images']} 张，其中有框 {stats['with_label']} 张")
    print(f"  总框数 {stats['boxes']}  {stats['per_class']}")
    print(f"  输出 -> {os.path.abspath(args.out)}")
    print("  下一步：打开 preview.jpg 抽查，删掉 labels/ 里明显误检的 txt，再跑 tools/build_dataset.py")


if __name__ == "__main__":
    main()
