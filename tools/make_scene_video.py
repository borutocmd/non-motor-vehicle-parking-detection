"""用一张真实照片生成「缓慢推移」的模拟监控视频流。

用途：没有真实监控视频时，用它验证「检测 → ROI → 停留判定 → 告警截图」全链路，
     告警截图里是真实车辆与真实场景（而非色块）。

原理：对原图做缓变裁剪（Ken Burns 推近 + 平移），叠加轻微噪声与亮度起伏，
     输出固定帧率视频。目标全程停在画面内，因此停留计时会持续累加。

用法
----
python tools/make_scene_video.py --image 真实照片.jpg --out samples/scene_demo.mp4 \
    --seconds 12 --zoom 1.18 --fps 25
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--zoom", type=float, default=1.18, help="最大推近倍率")
    ap.add_argument("--pan", default="none", choices=["none", "h", "v", "hv"], help="平移方向")
    ap.add_argument("--size", default="1280x720", help="输出分辨率，如 1280x720")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"读不出图片: {args.image}")
    H0, W0 = img.shape[:2]

    out_w, out_h = [int(v) for v in args.size.lower().split("x")]
    # 先按输出比例裁剪，避免变形
    src_ratio = W0 / H0
    dst_ratio = out_w / out_h
    if src_ratio > dst_ratio:
        new_w = int(H0 * dst_ratio)
        x0 = (W0 - new_w) // 2
        base = img[:, x0:x0 + new_w]
    else:
        new_h = int(W0 / dst_ratio)
        y0 = (H0 - new_h) // 2
        base = img[y0:y0 + new_h, :]
    base = cv2.resize(base, (out_w, out_h), interpolation=cv2.INTER_AREA)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, out_h))
    total = int(args.seconds * args.fps)
    rng = np.random.default_rng(0)

    for i in range(total):
        p = i / max(1, total - 1)                 # 0 → 1
        z = 1.0 + (args.zoom - 1.0) * p
        cw, ch = int(out_w / z), int(out_h / z)

        if args.pan in ("h", "hv"):
            cx = int((out_w - cw) / 2 + (cw * 0.12) * (p - 0.5))
        else:
            cx = (out_w - cw) // 2
        if args.pan in ("v", "hv"):
            cy = int((out_h - ch) / 2 + (ch * 0.12) * (p - 0.5))
        else:
            cy = (out_h - ch) // 2
        cx = int(np.clip(cx, 0, out_w - cw))
        cy = int(np.clip(cy, 0, out_h - ch))

        crop = base[cy:cy + ch, cx:cx + cw]
        frame = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        # 轻微噪声 + 亮度起伏，模拟真实摄像头
        noise = rng.normal(0, 2.0, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        k = 1.0 + 0.02 * np.sin(2 * np.pi * i / 90.0)
        frame = np.clip(frame.astype(np.float32) * k, 0, 255).astype(np.uint8)

        cv2.putText(frame, "CAM-01", (out_w - 110, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (235, 235, 235), 2, cv2.LINE_AA)
        vw.write(frame)

    vw.release()
    print(f"已生成 {args.out}  {out_w}x{out_h} {args.fps}fps {args.seconds}s ({total} 帧)")
    print(f"来源图片: {args.image} ({W0}x{H0})")


if __name__ == "__main__":
    main()
