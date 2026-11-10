"""对比评测：预训练 YOLOv8n（COCO 80 类，重映射到两类）vs 微调模型。

为什么要自己写 mAP：
  预训练模型输出的是 COCO 类别编号（1=bicycle, 3=motorcycle），
  而我们的数据集是 0/1 两类，ultralytics 的 val 无法直接对齐；
  用同一份评测代码跑两个模型，数字才可比。

用法
----
python tools/eval_compare.py --data ../data/coco_yolo --imgsz 416
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCO_TO_OURS = {1: 0, 3: 1}     # 默认：COCO bicycle→0, motorcycle→1（可被 --remap 覆盖）
NAMES = ["bicycle", "motorcycle"]


def load_gt(label_path: str) -> list[tuple[int, float, float, float, float]]:
    out = []
    if not os.path.exists(label_path):
        return out
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                out.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out


def xywhn_to_xyxy(cx, cy, w, h, W_, H_):
    return ((cx - w / 2) * W_, (cy - h / 2) * H_, (cx + w / 2) * W_, (cy + h / 2) * H_)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9
    return inter / ua


def average_precision(recall, precision):
    """全点插值 AP（与 COCO 接近的简化版）。"""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(model, img_dir, lbl_dir, imgsz, remap: bool, conf_thr=0.001, iou_thr=0.5):
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    preds = {c: [] for c in range(len(NAMES))}
    gt_count = {c: 0 for c in range(len(NAMES))}

    for fn in files:
        img = cv2.imread(os.path.join(img_dir, fn))
        if img is None:
            continue
        H_, W_ = img.shape[:2]
        gts = load_gt(os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt"))
        gt_boxes = {c: [] for c in range(len(NAMES))}
        for cid, cx, cy, w, h in gts:
            if cid in gt_boxes:
                gt_boxes[cid].append(xywhn_to_xyxy(cx, cy, w, h, W_, H_))
                gt_count[cid] += 1

        r = model.predict(img, imgsz=imgsz, conf=conf_thr, iou=0.7, verbose=False)[0]
        det_by_class = {c: [] for c in range(len(NAMES))}
        if r.boxes is not None:
            for b, c, p in zip(r.boxes.xyxy.cpu().numpy(),
                               r.boxes.cls.cpu().numpy().astype(int),
                               r.boxes.conf.cpu().numpy()):
                cid = COCO_TO_OURS.get(int(c), -1) if remap else int(c)
                if cid in det_by_class:
                    det_by_class[cid].append((list(map(float, b)), float(p)))

        for c in range(len(NAMES)):
            dets = sorted(det_by_class[c], key=lambda x: -x[1])
            used = [False] * len(gt_boxes[c])
            for box, score in dets:
                best_iou, best_j = 0.0, -1
                for j, g in enumerate(gt_boxes[c]):
                    v = iou(box, g)
                    if v > best_iou:
                        best_iou, best_j = v, j
                tp = best_iou >= iou_thr and best_j >= 0 and not used[best_j]
                if tp:
                    used[best_j] = True
                preds[c].append((score, 1 if tp else 0))

    aps = {}
    for c in range(len(NAMES)):
        if gt_count[c] == 0:
            continue
        arr = sorted(preds[c], key=lambda x: -x[0])
        if not arr:
            aps[NAMES[c]] = 0.0
            continue
        tp = np.cumsum([a[1] for a in arr])
        fp = np.cumsum([1 - a[1] for a in arr])
        recall = tp / (gt_count[c] + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        aps[NAMES[c]] = average_precision(recall, precision)
    return aps, gt_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(W, "..", "data", "coco_yolo"))
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--names", default="bicycle,motorcycle", help="数据集类别名，逗号分隔")
    ap.add_argument("--remap", default="", help="预训练模型的 COCO 类别→数据集类别映射，如 3=0 或 1=0,3=1")
    ap.add_argument("--weights", nargs="*", default=None, help="要对比的权重（默认两个内置）")
    args = ap.parse_args()

    global NAMES, COCO_TO_OURS
    NAMES = [s.strip() for s in args.names.split(",") if s.strip()]
    if args.remap:
        COCO_TO_OURS = {}
        for part in args.remap.split(","):
            a, b = part.split("=")
            COCO_TO_OURS[int(a)] = int(b)

    # 兼容两种布局：images/val+labels/val  或  valid/images+valid/labels
    candidates = [
        (os.path.join(args.data, "images", "val"), os.path.join(args.data, "labels", "val")),
        (os.path.join(args.data, "valid", "images"), os.path.join(args.data, "valid", "labels")),
        (os.path.join(args.data, "val", "images"), os.path.join(args.data, "val", "labels")),
        (os.path.join(args.data, "test", "images"), os.path.join(args.data, "test", "labels")),
    ]
    img_dir = lbl_dir = None
    for i, l in candidates:
        if os.path.isdir(i):
            img_dir, lbl_dir = i, l
            break
    if not img_dir:
        raise SystemExit(f"在 {args.data} 下找不到验证集目录（试过 images/val、valid/images 等）")
    print(f"验证集: {img_dir}  ({len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])} 张)")

    from ultralytics import YOLO

    if args.weights:
        runs = [(os.path.basename(w), w, ("remap" if i == 0 else "model")) for i, w in enumerate(args.weights)]
        runs = [(n, p, True if t == "remap" else False) for n, p, t in runs]
    else:
        runs = [
            ("预训练 YOLOv8n（COCO 80 类，重映射）", os.path.join(W, "weights", "yolov8n.pt"), True),
            ("微调 YOLOv8n（本项目，2 类）", os.path.join(W, "weights", "ebike_finetuned.pt"), False),
        ]

    rows = []
    for name, wpath, remap in runs:
        if not os.path.exists(wpath):
            print(f"跳过（缺权重）: {name}")
            continue
        print(f"\n>>> {name}")
        model = YOLO(wpath)
        aps, gt_count = evaluate(model, img_dir, lbl_dir, args.imgsz, remap)
        mean_ap = sum(aps.values()) / len(aps) if aps else 0.0
        rows.append((name, aps, mean_ap))
        for c, v in aps.items():
            print(f"    {c:12} AP@0.5 = {v:.4f}   (GT {gt_count[NAMES.index(c)]} 个)")
        print(f"    {'mAP@0.5':12} = {mean_ap:.4f}")

    print("\n" + "=" * 78)
    head = f"{'模型':<34}" + "".join(f"{n:>12}" for n in NAMES) + f"{'mAP50':>10}"
    print(head)
    print("=" * 78)
    for name, aps, m in rows:
        line = f"{name:<34}" + "".join(f"{aps.get(n, 0):>12.4f}" for n in NAMES) + f"{m:>10.4f}"
        print(line)
    print("=" * 78)


if __name__ == "__main__":
    main()
