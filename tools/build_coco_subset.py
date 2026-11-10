"""从 COCO 2017 zip 抽 bicycle / motorcycle 子集，转成 YOLO 格式。

支持两种图像包：
  * 完整 zip（zipfile 正常读取）
  * **被截断的 zip**（只下前 1~2 GB）——流式解析本地文件头，按需取图

数据来源：ModelScope 上的 COCO2017 镜像（国内直连 ~10 MB/s）
  annotations_trainval2017.zip   241 MB
  val2017.zip                    778 MB
  train2017.zip                  18.4 GB（可只下前 2 GB）

用法
----
# 完整包
python tools/build_coco_subset.py --annotations D:/cv_data/annotations_trainval2017.zip \
    --images D:/cv_data/val2017.zip --split val2017 --out ../data/coco_yolo --max 900

# 部分包（只下了前 2 GB）
python tools/build_coco_subset.py --annotations D:/cv_data/annotations_trainval2017.zip \
    --images D:/cv_data/train2017_partial.zip --split train2017 --partial \
    --out D:/cv_data/coco_train --max 1500
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import struct
import zipfile

COCO_BIKE, COCO_MOTO = 2, 4
CLASSES = ["bicycle", "motorcycle"]


class PartialZip:
    """读取被截断 zip 的前半段（COCO 条目是 stored，可逐条跳过）。"""

    SIG = 0x04034B50

    def __init__(self, path: str):
        self.path = path
        self.entries: dict[str, tuple[int, int]] = {}
        self._scan()

    def _scan(self):
        with open(self.path, "rb") as f:
            f.seek(0, os.SEEK_END)
            total = f.tell()
            pos = 0
            while pos + 30 <= total:
                f.seek(pos)
                head = f.read(30)
                if len(head) < 30:
                    break
                (sig, _ver, _flags, _comp, _mt, _md, _crc,
                 csize, _usize, fnl, exl) = struct.unpack("<IHHHHHIIIHH", head)
                if sig != self.SIG:
                    break
                name = f.read(fnl).decode("utf-8", "replace")
                data_off = pos + 30 + fnl + exl
                if csize > 0:
                    self.entries[name] = (data_off, csize)
                pos = data_off + csize

    def names(self) -> set[str]:
        return set(self.entries)

    def read(self, name: str) -> bytes:
        off, size = self.entries[name]
        with open(self.path, "rb") as f:
            f.seek(off)
            return f.read(size)


def load_annotations(zip_path: str, split: str, work: str) -> dict:
    member = f"annotations/instances_{split}.json"
    out = os.path.join(work, f"instances_{split}.json")
    if not os.path.exists(out):
        os.makedirs(work, exist_ok=True)
        print(f"[annot] 解压 {member} …")
        with zipfile.ZipFile(zip_path) as z, z.open(member) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
    else:
        print(f"[annot] 复用 {out}")
    with open(out, "r", encoding="utf-8") as f:
        return json.load(f)


def select(data: dict, max_images: int, min_area_ratio: float, seed: int = 42,
           present: set | None = None, split: str = "val2017"):
    imgs = {im["id"]: im for im in data["images"]}
    by_img: dict[int, list] = {}
    for ann in data["annotations"]:
        if ann["category_id"] not in (COCO_BIKE, COCO_MOTO) or ann.get("iscrowd"):
            continue
        x, y, w, h = ann["bbox"]
        if w <= 2 or h <= 2:
            continue
        by_img.setdefault(ann["image_id"], []).append((ann["category_id"], [x, y, w, h]))

    items, dropped = [], 0
    for iid, anns in by_img.items():
        im = imgs.get(iid)
        if not im:
            continue
        W, H = im["width"], im["height"]
        good = [a for a in anns if (a[1][2] * a[1][3]) / (W * H) >= min_area_ratio]
        if good:
            items.append({"id": iid, "file": im["file_name"], "w": W, "h": H, "anns": good})
        else:
            dropped += 1

    if present is not None:
        before = len(items)
        items = [it for it in items if f"{split}/{it['file']}" in present]
        print(f"[select] 与图像包求交集：{before} → {len(items)} 张")
    print(f"[select] 含 bicycle/motorcycle 的图 {len(by_img)} 张 → 过滤小目标后 {len(items)} 张")
    random.Random(seed).shuffle(items)
    if max_images and len(items) > max_images:
        items = items[:max_images]
        print(f"[select] 截取 {len(items)} 张")
    return items, {"candidate_images": len(by_img), "dropped_small": dropped}


def to_yolo(cid: int, bbox, W: int, H: int) -> str | None:
    x, y, w, h = bbox
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(W), x + w), min(float(H), y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
    nw, nh = (x2 - x1) / W, (y2 - y1) / H
    return f"{0 if cid == COCO_BIKE else 1} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val2017")
    ap.add_argument("--max", type=int, default=900)
    ap.add_argument("--min-area", type=float, default=0.002)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--partial", action="store_true", help="images 是截断的 zip")
    args = ap.parse_args()

    work = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_meta")
    data = load_annotations(args.annotations, args.split, work)

    if args.partial:
        zimg = PartialZip(args.images)
        names = set(zimg.entries)
        print(f"[images] 部分 zip：{len(names)} 个完整条目")
    else:
        zimg = zipfile.ZipFile(args.images)
        names = set(zimg.namelist())
        print(f"[images] 完整 zip：{len(names)} 个条目")

    items, sel_stats = select(data, args.max, args.min_area,
                              present=names, split=args.split)

    random.Random(7).shuffle(items)
    n_val = max(1, int(len(items) * args.val_ratio))
    parts = {"val": items[:n_val], "train": items[n_val:]}

    if args.partial:
        def read_member(name: str) -> bytes:
            return zimg.read(name)
    else:
        def read_member(name: str) -> bytes:
            with zimg.open(name) as s:
                return s.read()

    stats = {"source": f"COCO 2017 {args.split}" + (" (partial zip)" if args.partial else ""),
             "classes": CLASSES, "counts": {}, "boxes": 0,
             "per_class": {c: 0 for c in CLASSES}, **sel_stats}

    for part, subset in parts.items():
        di = os.path.join(args.out, "images", part)
        dl = os.path.join(args.out, "labels", part)
        os.makedirs(di, exist_ok=True)
        os.makedirs(dl, exist_ok=True)
        n_ok = missing = 0
        for it in subset:
            member = f"{args.split}/{it['file']}"
            if member not in names:
                missing += 1
                continue
            dst = os.path.join(di, it["file"])
            with open(dst, "wb") as f:
                f.write(read_member(member))
            lines = [to_yolo(c, b, it["w"], it["h"]) for c, b in it["anns"]]
            lines = [x for x in lines if x]
            if not lines:
                os.remove(dst)
                continue
            with open(os.path.join(dl, it["file"].replace(".jpg", ".txt")), "w",
                      encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            n_ok += 1
            stats["boxes"] += len(lines)
            for ln in lines:
                stats["per_class"][CLASSES[int(ln[0])]] += 1
        stats["counts"][part] = n_ok
        print(f"[write] {part}: {n_ok} 张（不在包内跳过 {missing}）")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(CLASSES) + "\n")
    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(args.out).replace(os.sep, '/')}\n"
                f"train: images/train\nval: images/val\n"
                f"names:\n  0: bicycle\n  1: motorcycle\n")
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n[done]")
    print(f"  训练 {stats['counts'].get('train', 0)} / 验证 {stats['counts'].get('val', 0)}")
    print(f"  框 {stats['boxes']}  {stats['per_class']}")
    print(f"  输出 {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
