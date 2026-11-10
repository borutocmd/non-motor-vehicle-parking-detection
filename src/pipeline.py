"""主流水线：读帧 → 检测 → 跟踪 → 区域入侵/违停判定 → 可视化 → 取证落盘。"""
from __future__ import annotations

import os
import time

import cv2

from .detector import build_detector
from .judge import KIND_INTRUSION, KIND_VIOLATION, ViolationJudge
from .roi import draw_zone, load_zones
from .tracker import IoUTracker
from .visual import draw_alarm_banner, draw_detections, draw_panel
from .writer import ViolationWriter


def _open_capture(source):
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap, True
    return cv2.VideoCapture(source), False


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.detector = build_detector(cfg)
        self.tracker = IoUTracker(iou_threshold=float(cfg.tracker.iou_threshold),
                                  max_age=int(cfg.tracker.max_age))
        alarm_ids = cfg.judge.get("alarm_class_ids")
        self.judge = ViolationJudge(
            dwell_seconds=float(cfg.judge.dwell_seconds),
            min_hits=int(cfg.judge.min_hits),
            cooldown_seconds=float(cfg.judge.cooldown_seconds),
            require_inside=bool(cfg.judge.require_inside),
            anchor=str(cfg.judge.anchor),
            mode=str(cfg.judge.get("mode", "both")),
            intrusion_hits=int(cfg.judge.get("intrusion_hits", 2)),
            alarm_class_ids=list(alarm_ids) if alarm_ids else None,
        )
        zones = load_zones(cfg.path(cfg.roi.polygons_file))
        self.zone = zones.get(str(cfg.roi.active))
        if self.zone is None and zones:
            self.zone = next(iter(zones.values()))
        self.colors = {
            "ok": tuple(cfg.visual.color_ok),
            "warn": tuple(cfg.visual.color_warn),
            "intrusion": tuple(cfg.visual.get("color_intrusion", [0, 120, 255])),
            "bad": tuple(cfg.visual.color_bad),
            "roi": tuple(cfg.visual.color_roi),
        }
        self.out_dir = cfg.path(cfg.output.dir)
        os.makedirs(self.out_dir, exist_ok=True)

    def run(self, source=None, max_frames: int = 0, show: bool = True,
            save_video: bool = False, video_out: str | None = None,
            on_frame=None, stop_flag=None) -> dict:
        cfg = self.cfg
        src = source if source is not None else cfg.path(str(cfg.input.source))
        cap, is_cam = _open_capture(src)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {src}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 0
        print(f"[input] {src}  {w}x{h}  {fps:.1f}fps  frames={total or 'live'}")
        print(f"[mode ] judge.mode={self.judge.mode}  停留阈值={self.judge.dwell_seconds:.0f}s  "
              f"告警类别={sorted(self.judge.alarm_class_ids) if self.judge.alarm_class_ids else '全部'}")

        vw = None
        if save_video:
            path = video_out or os.path.join(self.out_dir, "annotated.mp4")
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            print(f"[output] 标注视频 -> {path}")

        writer = ViolationWriter(
            self.out_dir,
            prefix=str(cfg.output.screenshot_prefix),
            save_csv=bool(cfg.output.save_csv),
            csv_name=str(cfg.output.csv_name),
            save_screenshot=bool(cfg.output.save_screenshot),
        )
        print(f"[output] 取证目录 -> {self.out_dir}")

        idx = 0
        t0 = time.time()
        alarm_until = 0.0
        alarm_text = ""
        alarm_color = self.colors["intrusion"]
        n_intrusion = n_violation = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    if cfg.input.loop and not is_cam:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        idx = 0
                        continue
                    break

                ts = idx / fps if fps > 0 else (time.time() - t0)
                dets = [{"box": d.box, "cls_id": d.cls_id, "conf": d.conf}
                        for d in self.detector.detect(frame)]
                tracks = self.tracker.update(dets, ts)
                res = self.judge.update(tracks, self.zone, w, h, ts, now=time.time())

                if self.zone is not None:
                    draw_zone(frame, self.zone, color=self.colors["roi"],
                              font_scale=float(cfg.visual.font_scale))
                draw_detections(frame, tracks, res, self.detector.names, self.colors)

                for ev in res.events:
                    cls_name = self.detector.names.get(ev.cls_id, str(ev.cls_id))
                    kind_cn = "区域入侵" if ev.kind == KIND_INTRUSION else "违停"
                    path = writer.save(frame, ev, cls_name, ts, kind_cn)
                    print(f"[{kind_cn}] #{ev.tid} {cls_name} dwell={ev.dwell:.1f}s -> {os.path.basename(path)}")
                    if ev.kind == KIND_INTRUSION:
                        n_intrusion += 1
                        alarm_text = f"非机动车进入禁停区：{cls_name} #{ev.tid}（{ev.zone}）"
                        alarm_color = self.colors["intrusion"]
                    else:
                        n_violation += 1
                        alarm_text = f"违停告警：{cls_name} #{ev.tid} 已停 {ev.dwell:.1f}s（{ev.zone}）"
                        alarm_color = self.colors["bad"]
                    alarm_until = time.time() + 2.5

                in_zone = sum(1 for s in res.status.values() if s != "outside")
                elapsed = time.time() - t0
                real_fps = idx / elapsed if elapsed > 0 else 0
                lines = [
                    f"帧 {idx}  实时 {real_fps:.1f} fps",
                    f"目标 {len(tracks)}  区内 {in_zone}",
                    f"入侵告警 {n_intrusion}   违停 {n_violation}",
                    f"停留阈值 {self.judge.dwell_seconds:.0f}s",
                ]
                alarm_on = alarm_until > time.time() and bool(alarm_text)
                draw_panel(frame, lines, origin=(10, 54 if alarm_on else 10),
                           size=int(16 * float(cfg.visual.font_scale) / 0.55 or 16))

                if alarm_on:
                    draw_alarm_banner(frame, alarm_text, color=alarm_color)

                if vw:
                    vw.write(frame)

                if show and bool(cfg.output.show_window):
                    disp = frame
                    sc = float(cfg.output.window_scale or 1.0)
                    if sc != 1.0:
                        disp = cv2.resize(frame, None, fx=sc, fy=sc)
                    cv2.imshow("Non-motor Vehicle Parking Violation", disp)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (27, ord("q")):
                        print("[stop] 用户中断")
                        break

                if on_frame is not None:
                    try:
                        on_frame(idx, total, frame, res, {"intrusion": n_intrusion,
                                                          "violations": n_violation})
                    except Exception as exc:      # 回调异常不应中断流水线
                        print("[warn] on_frame 回调异常:", exc)

                if stop_flag is not None and stop_flag():
                    print("[stop] 收到中断请求")
                    break

                idx += 1
                if max_frames and idx >= max_frames:
                    break
        finally:
            writer.close()
            cap.release()
            if vw:
                vw.release()
            if show:
                cv2.destroyAllWindows()

        stats = {"frames": idx, "intrusion": n_intrusion, "violations": n_violation,
                 "out_dir": self.out_dir, "csv": writer.csv_path}
        print(f"[done] 处理 {idx} 帧，入侵告警 {n_intrusion} 次，违停 {n_violation} 次")
        return stats
