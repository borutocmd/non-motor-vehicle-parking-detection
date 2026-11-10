"""从视频抽帧，作为自建数据集的原始素材。

用法
----
python tools/extract_frames.py --video ../../我的视频.mp4 --out datasets/raw_frames --every 15
python tools/extract_frames.py --video a.mp4 --out datasets/raw_frames --every 10 --max 400 --dedup 0.97

去重原理：把画面缩到 64x64 灰度图，和上一张保留帧比较像素差均值，
差异小于阈值就跳过（固定机位下大量帧几乎一样，去重能省 70%+ 的标注量）。
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def frame_sig(frame) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=15, help="每隔多少帧取一张")
    ap.add_argument("--max", type=int, default=0, help="最多取多少张（0=不限）")
    ap.add_argument("--dedup", type=float, default=0.0,
                    help="0=不去重；否则为相似度阈值(0.97 表示差异<3%% 视为重复)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {args.video}  {fps:.1f}fps  共 {total} 帧")

    saved = 0
    idx = 0
    last_sig = None
    base = os.path.splitext(os.path.basename(args.video))[0]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % args.every == 0:
            keep = True
            if args.dedup > 0 and last_sig is not None:
                cur = frame_sig(frame)
                diff = float(np.mean(np.abs(cur - last_sig)) / 255.0)
                keep = diff > (1.0 - args.dedup)
            if keep:
                name = f"{base}_{idx:06d}.jpg"
                path = os.path.join(args.out, name)
                from src.writer import _imwrite_unicode  # 兼容中文路径
                _imwrite_unicode(path, frame)
                last_sig = frame_sig(frame)
                saved += 1
                if args.max and saved >= args.max:
                    break
        idx += 1

    cap.release()
    print(f"[done] 累计读 {idx} 帧，保存 {saved} 张 -> {os.path.abspath(args.out)}")
    if saved == 0:
        print("⚠ 没抽到帧，检查 --every 是否大于视频总帧数")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
