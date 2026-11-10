"""数据集清洗：删除不相关图片与近似重复帧。

判据
----
1. **无标签 / 空标签**         → 没有训练价值
2. **竖屏截图**（内容宽度占比 < 0.6） → 手机截图/相册截图，Roboflow 补黑边导致两侧大片空白
3. **横屏全景**（内容高度占比 < 0.6） → 超宽长图，通常是页面截图
4. **白底商品图**（内容区近白像素 > 60%）→ 电商详情图，不是真实场景
5. **标注异常**（单框面积 > 90%）        → 框满整图，错标
6. **目标过小**（全部框面积 < 0.3%）     → 远处小点，监控场景学不到东西
7. **近似重复**（dHash 汉明距离 ≤ 6）    → 同一场景连拍，去重保留第一张

用法
----
python tools/clean_dataset.py --src D:/cv_data/datasets/motorcyclemodels.v1i.yolov8 \
    --out D:/cv_data/cleaned/motorcyclemodels --preview D:/cv_data/clean_preview
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict

import cv2
import numpy as np


def dhash(img, size=8) -> int:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def content_box(img, thresh: float = 6.0):
    """找非补边的内容区域（Roboflow letterbox 会补纯色边）。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    col_std = g.std(axis=0)
    row_std = g.std(axis=1)
    cols = np.where(col_std > thresh)[0]
    rows = np.where(row_std > thresh)[0]
    if len(cols) < 4 or len(rows) < 4:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def white_ratio(img, box) -> float:
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    mn = crop.min(axis=2)
    return float((mn > 230).mean())


def flat_row_ratio(img, thr: float = 3.0) -> float:
    """平坦行占比：整行像素几乎无变化 → UI 栏/纯色背景，截图特征。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float((g.std(axis=1) < thr).mean())


def load_labels(path: str):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                out.append((int(float(p[0])), float(p[3]), float(p[4])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", default=None, help="把被删的图拼成预览")
    ap.add_argument("--dup-thresh", type=int, default=6)
    ap.add_argument("--min-content-ratio", type=float, default=0.60)
    ap.add_argument("--white-thresh", type=float, default=0.60)
    ap.add_argument("--huge-box", type=float, default=0.90)
    ap.add_argument("--tiny-box", type=float, default=0.003)
    ap.add_argument("--flat-rows", type=float, default=0.25,
                    help="平坦行占比超过该值视为截图（照片通常 <0.15）")
    args = ap.parse_args()

    src = args.src
    out = args.out
    if os.path.isdir(out):
        shutil.rmtree(out)

    # 收集 (part, img_rel, lbl_rel)
    pairs = []
    for root, _, files in os.walk(src):
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_p = os.path.join(root, f)
            rel = os.path.relpath(img_p, src)
            part = rel.split(os.sep)[0] if os.sep in rel else "."
            lbl_p = os.path.join(os.path.dirname(root), "labels", f.rsplit(".", 1)[0] + ".txt")
            if not os.path.exists(lbl_p):
                alt = os.path.join(root.replace("images", "labels"), f.rsplit(".", 1)[0] + ".txt")
                lbl_p = alt if os.path.exists(alt) else ""
            pairs.append((part, img_p, lbl_p, rel))

    print(f"扫描 {len(pairs)} 张图")

    removed = []          # (path, reason)
    kept = []
    seen_hashes = defaultdict(list)   # part -> [(hash, path)]
    per_reason = Counter()

    for part, img_p, lbl_p, rel in pairs:
        img = cv2.imread(img_p)
        if img is None:
            removed.append((img_p, "读不出"))
            per_reason["读不出"] += 1
            continue
        H, W = img.shape[:2]
        boxes = load_labels(lbl_p) if lbl_p else []

        if not boxes:
            removed.append((img_p, "无标签"))
            per_reason["无标签"] += 1
            continue

        areas = [w * h for _, w, h in boxes]
        if max(areas) > args.huge_box:
            removed.append((img_p, "框满整图"))
            per_reason["框满整图"] += 1
            continue
        if max(areas) < args.tiny_box:
            removed.append((img_p, "目标过小"))
            per_reason["目标过小"] += 1
            continue

        cb = content_box(img)
        if cb:
            cw, ch = cb[2] - cb[0], cb[3] - cb[1]
            if cw / W < args.min_content_ratio:
                removed.append((img_p, "竖屏截图"))
                per_reason["竖屏截图"] += 1
                continue
            if ch / H < args.min_content_ratio:
                removed.append((img_p, "横屏长图"))
                per_reason["横屏长图"] += 1
                continue
            if white_ratio(img, cb) > args.white_thresh:
                removed.append((img_p, "白底商品图"))
                per_reason["白底商品图"] += 1
                continue

        fr = flat_row_ratio(img)
        if fr > args.flat_rows:
            removed.append((img_p, "截图/UI"))
            per_reason["截图/UI"] += 1
            continue

        h = dhash(img)
        dup = None
        for prev_h, prev_p in seen_hashes[part]:
            if hamming(h, prev_h) <= args.dup_thresh:
                dup = prev_p
                break
        if dup:
            removed.append((img_p, "重复"))
            per_reason["重复"] += 1
            continue
        seen_hashes[part].append((h, img_p))
        kept.append((part, img_p, lbl_p, rel))

    # 写出保留的
    for part, img_p, lbl_p, rel in kept:
        dst_img = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        shutil.copy2(img_p, dst_img)
        if lbl_p:
            dst_lbl = os.path.join(os.path.dirname(os.path.dirname(dst_img)), "labels",
                                   os.path.basename(lbl_p))
            os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
            shutil.copy2(lbl_p, dst_lbl)

    # 拷 data.yaml 等
    for extra in ("data.yaml", "data.yml", "classes.txt", "README.dataset.txt", "README.roboflow.txt"):
        p = os.path.join(src, extra)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out, extra))

    print(f"\n保留 {len(kept)} 张，删除 {len(removed)} 张")
    for r, c in per_reason.most_common():
        print(f"   {r:8} {c:5} 张")
    by_part = Counter(p for p, *_ in kept)
    print(f"   保留分布: {dict(by_part)}")

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "_clean_report.json"), "w", encoding="utf-8") as f:
        json.dump({"src": src, "kept": len(kept), "removed": len(removed),
                   "reasons": dict(per_reason),
                   "removed_list": [(os.path.relpath(p, src), r) for p, r in removed],
                   "kept_splits": dict(by_part)}, f, ensure_ascii=False, indent=2)

    # 被删图片预览
    if args.preview and removed:
        os.makedirs(args.preview, exist_ok=True)
        cell = (300, 170)
        cols = 5
        picks = removed[:40]
        rows = (len(picks) + cols - 1) // cols
        sheet = np.full((rows * cell[1], cols * cell[0], 3), 245, np.uint8)
        for i, (p, reason) in enumerate(picks):
            im = cv2.imread(p)
            if im is None:
                continue
            h, w = im.shape[:2]
            s = min(cell[0] / w, cell[1] / h)
            th = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
            r_, c_ = divmod(i, cols)
            y0, x0 = r_ * cell[1], c_ * cell[0]
            sheet[y0:y0 + th.shape[0], x0:x0 + th.shape[1]] = th
            cv2.rectangle(sheet, (x0, y0), (x0 + cell[0] - 1, y0 + 16), (40, 40, 40), -1)
            cv2.putText(sheet, reason, (x0 + 4, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 255, 255), 1, cv2.LINE_AA)
        name = os.path.basename(src.rstrip("\\/")) + "_removed.jpg"
        ok, enc = cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            with open(os.path.join(args.preview, name), "wb") as f:
                f.write(enc.tobytes())
            print(f"   被删预览 -> {os.path.join(args.preview, name)}")


if __name__ == "__main__":
    main()
