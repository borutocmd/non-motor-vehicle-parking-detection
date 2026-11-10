"""生成成果截图：用真实场景照片产出「检测 / 区域入侵 / 违停」三张系统运行截图。

不依赖视频：直接对静态图片跑完整判定链路（构造时序状态触发两层告警），
并把画面快照定格在**事件发生的那一帧**，保证横幅文字与框上秒数一致。

用法
----
python tools/make_result_screenshots.py --image 照片.jpg --out ../outputs/report_figs \
    --zone 0.22,0.12,0.93,0.10,0.95,0.80,0.30,0.88
（--zone 给 4 个数 x1,y1,x2,y2 为矩形；给 8 个数为四边形多边形）
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config          # noqa: E402
from src.detector import build_detector      # noqa: E402
from src.judge import ViolationJudge         # noqa: E402
from src.roi import RoiZone, draw_zone       # noqa: E402
from src.tracker import IoUTracker           # noqa: E402
from src.visual import draw_alarm_banner, draw_detections, draw_panel  # noqa: E402
from src.writer import _imwrite_unicode      # noqa: E402

COLORS = {"ok": (80, 220, 100), "warn": (0, 165, 255),
          "intrusion": (0, 120, 255), "bad": (50, 50, 235)}


def zone_from_arg(arg: str) -> RoiZone | None:
    if not arg:
        return None
    vals = [float(v) for v in arg.split(",")]
    if len(vals) == 4:
        x1, y1, x2, y2 = vals
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    elif len(vals) == 8:
        pts = [(vals[i], vals[i + 1]) for i in range(0, 8, 2)]
    else:
        raise SystemExit("--zone 需要 4 或 8 个数")
    return RoiZone(name="停车棚区域", points_norm=pts, normalized=True)


def auto_zone(w, h, boxes, pad=0.22) -> RoiZone:
    if not boxes:
        return RoiZone(name="停车棚区域",
                       points_norm=[(0.25, 0.30), (0.75, 0.30), (0.78, 0.88), (0.22, 0.88)],
                       normalized=True)
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad * 0.5
    y2 += bh * pad * 0.3
    nx1, nx2 = np.clip([x1 / w, x2 / w], 0.02, 0.98)
    ny1, ny2 = np.clip([y1 / h, y2 / h], 0.02, 0.98)
    return RoiZone(name="停车棚区域",
                   points_norm=[(nx1, ny1), (nx2, ny1), (nx2, ny2), (nx1, ny2)],
                   normalized=True)


def render(frame, zone, detector, judge, tracker, ts):
    h, w = frame.shape[:2]
    dets = [{"box": d.box, "cls_id": d.cls_id, "conf": d.conf} for d in detector.detect(frame)]
    tracks = tracker.update(dets, ts)
    res = judge.update(tracks, zone, w, h, ts)

    canvas = frame.copy()
    draw_zone(canvas, zone, color=(0, 215, 255), font_scale=0.55)
    draw_detections(canvas, tracks, res, detector.names, COLORS)
    in_zone = sum(1 for s in res.status.values() if s != "outside")
    panel = [
        f"镜头 CAM-01   帧时间 {ts:.1f}s",
        f"检测目标 {len(tracks)}   禁停区内 {in_zone}",
        f"入侵告警 {judge.stats['intrusion']}   违停 {judge.stats['violation']}",
        f"停留阈值 {judge.dwell_seconds:.0f}s",
    ]
    draw_panel(canvas, panel, origin=(10, 10), size=15)
    return canvas, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--zone", default="")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(here, "config.yaml"))
    if args.weights:
        cfg._data["model"]["weights"] = args.weights
    detector = build_detector(cfg)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"读不出图片: {args.image}")
    h, w = img.shape[:2]

    probe = [d.box for d in detector.detect(img)]
    zone = zone_from_arg(args.zone) or auto_zone(w, h, probe)

    os.makedirs(args.out, exist_ok=True)
    base = args.prefix or os.path.splitext(os.path.basename(args.image))[0]

    # ① 只检测，不告警
    j0 = ViolationJudge(dwell_seconds=999, min_hits=999, mode="dwell", alarm_class_ids=[])
    canvas0, _ = render(img, zone, detector, j0, IoUTracker(0.3, 30), 0.0)
    p1 = os.path.join(args.out, f"fig1_detect_{base}.jpg")
    _imwrite_unicode(p1, canvas0)
    print("① 检测结果 ->", os.path.basename(p1))

    # ② 区域入侵：定格在告警那一帧
    j1 = ViolationJudge(dwell_seconds=8.0, min_hits=5, intrusion_hits=2,
                        mode="both", alarm_class_ids=[1, 3])
    trk1 = IoUTracker(0.3, 30)
    shot1 = None
    canvas1 = img.copy()
    for k in range(6):
        canvas1, res1 = render(img, zone, detector, j1, trk1, k * 0.5)
        if shot1 is None and res1.events:
            ev = res1.events[0]
            nm = detector.names.get(ev.cls_id, str(ev.cls_id))
            draw_alarm_banner(canvas1, f"非机动车进入禁停区：{nm} #{ev.tid}（{ev.zone}）",
                              color=COLORS["intrusion"])
            shot1 = canvas1.copy()
    p2 = os.path.join(args.out, f"fig2_intrusion_{base}.jpg")
    _imwrite_unicode(p2, shot1 if shot1 is not None else canvas1)
    print("② 区域入侵 ->", os.path.basename(p2))

    # ③ 违停：定格在升级为违停的那一帧
    j2 = ViolationJudge(dwell_seconds=8.0, min_hits=5, intrusion_hits=2,
                        mode="both", alarm_class_ids=[1, 3])
    trk2 = IoUTracker(0.3, 30)
    shot2 = None
    canvas2 = img.copy()
    for k in range(26):
        canvas2, res2 = render(img, zone, detector, j2, trk2, k * 0.5)
        if shot2 is None:
            for e in res2.events:
                if e.kind == "violation":
                    nm = detector.names.get(e.cls_id, str(e.cls_id))
                    draw_alarm_banner(
                        canvas2, f"违停告警：{nm} #{e.tid} 已停 {e.dwell:.1f}s（{e.zone}）",
                        color=COLORS["bad"])
                    shot2 = canvas2.copy()
                    break
    p3 = os.path.join(args.out, f"fig3_violation_{base}.jpg")
    _imwrite_unicode(p3, shot2 if shot2 is not None else canvas2)
    print("③ 违停告警 ->", os.path.basename(p3))

    print(f"\n统计：入侵 {j2.stats['intrusion']} 次，违停 {j2.stats['violation']} 次")


if __name__ == "__main__":
    main()
