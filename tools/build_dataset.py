"""把标注好的图片+标签整理成 ultralytics 可直接训练的目录结构。

用法
----
python tools/build_dataset.py --src datasets/labeled --out D:/cv_data/campus_yolo --val-ratio 0.15

产出
----
<out>/images/train/*.jpg   <out>/labels/train/*.txt
<out>/images/val/*.jpg     <out>/labels/val/*.txt
<out>/data.yaml            # 直接给 yolo train 用
"""
from __future__ import annotations

import argparse
import os
import random
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="含 images/ 和 labels/ 的目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--names", default="bicycle,motorcycle,person")
    args = ap.parse_args()

    src_img = os.path.join(args.src, "images")
    src_lbl = os.path.join(args.src, "labels")
    if not os.path.isdir(src_img):
        raise SystemExit(f"找不到 {src_img}")

    files = []
    for fn in sorted(os.listdir(src_img)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(fn)[0]
        lbl = os.path.join(src_lbl, stem + ".txt")
        if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
            files.append((fn, lbl))
    if not files:
        raise SystemExit("没有可用的「有标签」样本，先跑 autolabel.py")

    random.Random(args.seed).shuffle(files)
    n_val = max(1, int(len(files) * args.val_ratio))
    split = {"val": files[:n_val], "train": files[n_val:]}

    for part, items in split.items():
        di = os.path.join(args.out, "images", part)
        dl = os.path.join(args.out, "labels", part)
        os.makedirs(di, exist_ok=True)
        os.makedirs(dl, exist_ok=True)
        for fn, lbl in items:
            shutil.copy2(os.path.join(src_img, fn), os.path.join(di, fn))
            shutil.copy2(lbl, os.path.join(dl, os.path.splitext(fn)[0] + ".txt"))
        print(f"  {part}: {len(items)} 张")

    names = [s.strip() for s in args.names.split(",") if s.strip()]
    lines = [f"path: {os.path.abspath(args.out).replace(os.sep, '/')}",
             "train: images/train", "val: images/val", "names:"]
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    yaml_path = os.path.join(args.out, "data.yaml")
    os.makedirs(args.out, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n数据集就绪 -> {os.path.abspath(args.out)}")
    print(f"配置: {yaml_path}")
    print("\n训练命令：")
    print(f"  yolo detect train data={yaml_path} model=weights/yolov8n.pt "
          f"epochs=50 imgsz=640 batch=8 device=cpu")


if __name__ == "__main__":
    main()
