"""命令行入口。

用法示例
--------
python main.py                                  # 用 config.yaml 的配置
python main.py --source samples/demo.mp4        # 指定视频
python main.py --mock                           # 无权重自测模式（颜色检测）
python main.py --source 0                       # 调用摄像头
python main.py --no-window --max-frames 300 --save-video
python main.py --make-roi 1280 720              # 交互式画禁停区并写入 configs/roi.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config
from src.pipeline import Pipeline


def parse_args():
    p = argparse.ArgumentParser(description="非机动车违停智能检测系统")
    p.add_argument("--config", default=None, help="配置文件路径")
    p.add_argument("--source", default=None, help="视频路径或摄像头编号")
    p.add_argument("--mock", action="store_true", help="用颜色检测器自测（无需权重）")
    p.add_argument("--no-window", action="store_true", help="不弹窗（无界面环境）")
    p.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧")
    p.add_argument("--save-video", action="store_true", help="导出标注后的视频")
    p.add_argument("--loop", action="store_true", help="视频循环播放")
    p.add_argument("--dwell", type=float, default=None, help="覆盖停留时长阈值（秒）")
    p.add_argument("--make-roi", nargs=2, type=int, metavar=("W", "H"),
                   help="交互式画禁停区，参数为画面宽高")
    return p.parse_args()


def make_roi(width: int, height: int):
    """摄像头画面上点选多边形，回车结束，写入 configs/roi.json（归一化坐标）。"""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头打不开")
        return
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("读帧失败")
        return

    frame = cv2.resize(frame, (width, height))
    pts: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    print("左键加点（>=3 个）｜右键撤销｜Enter 完成｜R 重画｜Q 退出")
    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI", on_mouse)
    while True:
        canvas = frame.copy()
        for p in pts:
            cv2.circle(canvas, p, 5, (0, 215, 255), -1)
        if len(pts) >= 3:
            cv2.polylines(canvas, [np.array(pts, np.int32)], True, (0, 215, 255), 2)
        cv2.imshow("ROI", canvas)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("r"):
            pts.clear()
        elif k in (13, 10):
            break
        elif k == ord("q"):
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()

    if len(pts) < 3:
        print("点数不足 3，取消")
        return

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "roi.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    zones = data.setdefault("zones", {})
    zones["lot_A"] = [[round(x / width, 4), round(y / height, 4)] for x, y in pts]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(pts)} 个点 -> {path}")


def main():
    args = parse_args()

    if args.make_roi:
        make_roi(*args.make_roi)
        return

    cfg = load_config(args.config)
    if args.mock:
        cfg._data["model"]["mock"] = True
    if args.loop:
        cfg._data["input"]["loop"] = True
    if args.dwell is not None:
        cfg._data["judge"]["dwell_seconds"] = float(args.dwell)
    if args.no_window:
        cfg._data["output"]["show_window"] = False

    source = args.source
    if source is not None and source.isdigit():
        source = int(source)

    pipe = Pipeline(cfg)
    pipe.run(source=source, max_frames=args.max_frames,
             show=not args.no_window, save_video=args.save_video)


if __name__ == "__main__":
    main()
