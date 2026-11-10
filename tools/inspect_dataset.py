"""数据集体检：结构 / 类别 / 框大小分布 / 预览拼图。

用法
----
python tools/inspect_dataset.py --zip 路径.zip --out D:/tmp/preview
python tools/inspect_dataset.py --dir 已解压目录 --out D:/tmp/preview

产出
----
<out>/<数据集名>_preview.jpg   12 张样本 + 画好的框（直接看内容对不对）
<out>/<数据集名>_stats.json    统计（图片数、类别、框大小分布）

判断标准（见 data/数据集挑选清单.md）
  * 平均框面积占比 < 2%      → 目标太小，监控场景里会漏检
  * 小框(<1%) 占比 > 30%     → 废图多
  * 类别名乱码/不连续        → 标注不规范
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import zipfile

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png")
CELL = (320, 180)   # 拼图每格 宽x高


def read_yaml_names(text: str) -> list[str]:
    """从 data.yaml 里抠出 names 列表（不用 yaml 库，容忍各种写法）。"""
    names = []
    in_names = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.strip().startswith("names:"):
            in_names = True
            tail = line.split(":", 1)[1].strip()
            if tail.startswith("["):
                items = tail.strip("[]").split(",")
                return [x.strip().strip("'\"") for x in items if x.strip()]
            continue
        if in_names:
            if ":" in line and line.startswith((" ", "\t", "-")):
                v = line.split(":", 1)[1].strip().strip("'\"")
                if v:
                    names.append(v)
            elif line.startswith("-"):
                names.append(line.lstrip("- ").strip().strip("'\""))
            elif not line.startswith((" ", "\t")):
                break
    return names


class Source:
    """统一封装 zip / 目录两种来源。"""

    def __init__(self, zip_path: str | None = None, dir_path: str | None = None):
        self.zip = zip_path
        self.dir = dir_path
        self._zf = zipfile.ZipFile(zip_path) if zip_path else None

    def namelist(self) -> list[str]:
        if self._zf:
            return self._zf.namelist()
        out = []
        for root, _, files in os.walk(self.dir):
            for f in files:
                out.append(os.path.relpath(os.path.join(root, f), self.dir).replace(os.sep, "/"))
        return out

    def read(self, name: str) -> bytes:
        if self._zf:
            return self._zf.read(name)
        with open(os.path.join(self.dir, name), "rb") as f:
            return f.read()

    def close(self):
        if self._zf:
            self._zf.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=12)
    args = ap.parse_args()

    if not args.zip and not args.dir:
        raise SystemExit("需要 --zip 或 --dir")

    src = Source(args.zip, args.dir)
    names_all = src.namelist()
    base = os.path.basename(args.zip or args.dir).replace(".zip", "")
    os.makedirs(args.out, exist_ok=True)

    # ---- 结构 ----
    yaml_files = [n for n in names_all if n.lower().endswith(("data.yaml", "data.yml"))]
    imgs = [n for n in names_all if n.lower().endswith(IMG_EXT)]
    txts = [n for n in names_all if n.lower().endswith(".txt")]

    print(f"\n{'='*58}")
    print(f"数据集: {base}")
    print(f"{'='*58}")
    print(f"  条目总数: {len(names_all)}  图片: {len(imgs)}  标签 txt: {len(txts)}")

    classes = []
    for y in yaml_files[:1]:
        txt = src.read(y).decode("utf-8", "replace")
        classes = read_yaml_names(txt)
        print(f"  data.yaml: {y}")
        print(f"  类别({len(classes)}): {classes}")
        for line in txt.splitlines():
            if line.strip().startswith(("nc:", "names:")):
                print(f"    {line.strip()[:80]}")

    # 划分统计
    splits = {}
    for n in imgs:
        for part in ("train", "valid", "test", "val"):
            if f"/{part}/" in n:
                splits[part] = splits.get(part, 0) + 1
                break
    print(f"  划分: {splits or '无明确划分'}")

    # ---- 标签统计 ----
    label_map = {}
    for n in txts:
        if n.lower().endswith("classes.txt"):
            continue
        label_map[n] = n  # 保留原路径，稍后按 stem 匹配

    def label_for(img_name: str) -> str | None:
        stem = os.path.splitext(img_name)[0]
        cand = stem.replace("/images/", "/labels/") + ".txt"
        if cand in set(txts):
            return cand
        stem_base = os.path.splitext(os.path.basename(img_name))[0]
        for t in txts:
            if os.path.splitext(os.path.basename(t))[0] == stem_base:
                return t
        return None

    per_class = {}
    areas = []
    tiny = 0
    total_boxes = 0
    per_image_boxes = []

    for im in imgs:
        lb = label_for(im)
        if not lb:
            continue
        content = src.read(lb).decode("utf-8", "replace")
        rows = [r.split() for r in content.splitlines() if len(r.split()) >= 5]
        per_image_boxes.append(len(rows))
        for r in rows:
            cid = int(float(r[0]))
            w, h = float(r[3]), float(r[4])
            a = w * h
            areas.append(a)
            total_boxes += 1
            name = classes[cid] if 0 <= cid < len(classes) else f"cls{cid}"
            per_class[name] = per_class.get(name, 0) + 1
            if a < 0.01:
                tiny += 1

    print(f"  标注框总数: {total_boxes}")
    print(f"  每类框数: {per_class}")
    if areas:
        arr = np.array(areas)
        print(f"  框面积占比: 平均 {arr.mean()*100:.2f}%  中位 {np.median(arr)*100:.2f}%  "
              f"最小 {arr.min()*100:.2f}%  最大 {arr.max()*100:.2f}%")
        print(f"  <1% 的小框: {tiny} 个 ({tiny/max(1,total_boxes)*100:.1f}%)")
        print(f"  每张图平均框数: {np.mean(per_image_boxes):.2f}")

    # ---- 预览拼图 ----
    sample = imgs[:]
    random.Random(0).shuffle(sample)
    sample = sample[: args.samples]
    cols = 3
    rows = (len(sample) + cols - 1) // cols
    sheet = np.full((rows * CELL[1], cols * CELL[0], 3), 245, np.uint8)

    for idx, im in enumerate(sample):
        try:
            buf = np.frombuffer(src.read(im), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            continue
        if img is None:
            continue
        H, W = img.shape[:2]
        lb = label_for(im)
        if lb:
            for r in src.read(lb).decode("utf-8", "replace").splitlines():
                p = r.split()
                if len(p) < 5:
                    continue
                cid = int(float(p[0]))
                cx, cy, w, h = map(float, p[1:5])
                x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
                x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
                color = (0, 200, 0) if cid % 2 == 0 else (0, 140, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                nm = classes[cid] if 0 <= cid < len(classes) else str(cid)
                cv2.putText(img, "%s %.1f%%" % (nm, w * h * 100), (x1 + 2, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        # 缩放铺格
        scale = min(CELL[0] / W, CELL[1] / H)
        thumb = cv2.resize(img, (max(1, int(W * scale)), max(1, int(H * scale))))
        r_, c_ = divmod(idx, cols)
        y0, x0 = r_ * CELL[1], c_ * CELL[0]
        sheet[y0:y0 + thumb.shape[0], x0:x0 + thumb.shape[1]] = thumb

    out_img = os.path.join(args.out, f"{base}_preview.jpg")
    ok, enc = cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if ok:
        with open(out_img, "wb") as f:
            f.write(enc.tobytes())
        print(f"  预览图 -> {out_img}")

    stats = {
        "dataset": base, "images": len(imgs), "labels": len(txts),
        "classes": classes, "splits": splits, "boxes": total_boxes,
        "per_class": per_class,
        "mean_area_pct": round(float(np.mean(areas)) * 100, 3) if areas else 0,
        "median_area_pct": round(float(np.median(areas)) * 100, 3) if areas else 0,
        "tiny_ratio_pct": round(tiny / max(1, total_boxes) * 100, 1),
        "boxes_per_image": round(float(np.mean(per_image_boxes)), 2) if per_image_boxes else 0,
    }
    with open(os.path.join(args.out, f"{base}_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    src.close()


if __name__ == "__main__":
    main()
