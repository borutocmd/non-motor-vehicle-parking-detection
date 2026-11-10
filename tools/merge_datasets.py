"""把多个 YOLO 数据集目录合并成一个，并重新按图片级划分 train/val。

用法
----
python tools/merge_datasets.py --srcs ../data/coco_yolo D:/cv_data/coco_train2017 \
    --out ../data/coco_yolo --val-ratio 0.15 --names bicycle,motorcycle

说明
----
* 每个 src 可以是任意已有的 YOLO 目录（含 images/train、images/val 等）
* 会去重（同名文件只保留一份）
* 先写到 <out>.tmp，成功后再替换 <out>，避免中途失败把原数据弄坏
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil

IMG_EXT = (".jpg", ".jpeg", ".png")


def collect(src: str) -> list[tuple[str, str]]:
    """返回 [(image_path, label_path), ...]，遍历 src/images/*"""
    out = []
    img_root = os.path.join(src, "images")
    if not os.path.isdir(img_root):
        return out
    for part in sorted(os.listdir(img_root)):
        di = os.path.join(img_root, part)
        if not os.path.isdir(di):
            continue
        dl = os.path.join(src, "labels", part)
        for fn in sorted(os.listdir(di)):
            if not fn.lower().endswith(IMG_EXT):
                continue
            lbl = os.path.join(dl, os.path.splitext(fn)[0] + ".txt")
            if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
                out.append((os.path.join(di, fn), lbl))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--names", default="bicycle,motorcycle")
    args = ap.parse_args()

    seen: dict[str, tuple[str, str]] = {}
    for src in args.srcs:
        pairs = collect(src)
        print(f"[src] {src}: {len(pairs)} 对")
        for img, lbl in pairs:
            key = os.path.basename(img)
            if key not in seen:          # 同名去重
                seen[key] = (img, lbl)

    pairs = list(seen.values())
    print(f"[merge] 去重后 {len(pairs)} 张")
    if not pairs:
        raise SystemExit("没有可用样本")

    random.Random(args.seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_ratio))
    split = {"val": pairs[:n_val], "train": pairs[n_val:]}

    tmp = args.out.rstrip("\\/") + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    for part, items in split.items():
        di = os.path.join(tmp, "images", part)
        dl = os.path.join(tmp, "labels", part)
        os.makedirs(di, exist_ok=True)
        os.makedirs(dl, exist_ok=True)
        for img, lbl in items:
            shutil.copy2(img, os.path.join(di, os.path.basename(img)))
            shutil.copy2(lbl, os.path.join(dl, os.path.basename(lbl)))
        print(f"[write] {part}: {len(items)} 张")

    names = [s.strip() for s in args.names.split(",") if s.strip()]
    boxes = 0
    per_class = {n: 0 for n in names}
    for part in split:
        dl = os.path.join(tmp, "labels", part)
        for fn in os.listdir(dl):
            with open(os.path.join(dl, fn), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    boxes += 1
                    cid = int(line.split()[0])
                    if 0 <= cid < len(names):
                        per_class[names[cid]] += 1

    with open(os.path.join(tmp, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")
    with open(os.path.join(tmp, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(tmp).replace(os.sep, '/')}\n"
                "train: images/train\nval: images/val\nnames:\n"
                + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    stats = {"sources": args.srcs, "images": len(pairs), "boxes": boxes,
             "per_class": per_class, "counts": {k: len(v) for k, v in split.items()}}
    with open(os.path.join(tmp, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 替换
    backup = args.out.rstrip("\\/") + ".bak"
    if os.path.isdir(args.out):
        shutil.move(args.out, backup)
    shutil.move(tmp, args.out)
    shutil.rmtree(backup, ignore_errors=True)

    # data.yaml 里的 path 改成最终路径
    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path, encoding="utf-8") as f:
        lines = f.readlines()
    lines[0] = f"path: {os.path.abspath(args.out).replace(os.sep, '/')}\n"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n[done] 合并完成 -> {os.path.abspath(args.out)}")
    print(f"  图片 {len(pairs)} 张（train {stats['counts']['train']} / val {stats['counts']['val']}）")
    print(f"  框 {boxes}  {per_class}")


if __name__ == "__main__":
    main()
