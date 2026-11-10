"""非机动车违停检测系统 · Web 演示后端（FastAPI）。

启动：
    python web/server.py            # 默认 http://127.0.0.1:8123

功能：
    * 列出示例视频 / 上传视频
    * 启动一次检测任务（可调参），后台线程跑真实流水线
    * 提供进度、实时预览帧、告警列表、截图、标注视频与 CSV
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

sys.path.insert(0, CODE)

from src.config import load_config          # noqa: E402
from src.pipeline import Pipeline            # noqa: E402
from src.writer import _imwrite_unicode      # noqa: E402

WEB_DIR = os.path.join(CODE, "web")
STATIC = os.path.join(WEB_DIR, "static")
RUNS = os.path.join(CODE, "outputs", "web")
UPLOADS = os.path.join(WEB_DIR, "uploads")
os.makedirs(RUNS, exist_ok=True)
os.makedirs(UPLOADS, exist_ok=True)

# 视频 → 禁停区 / 是否走演示模式
SAMPLE_ROI = {"parking_demo.mp4": "lot_A", "scene_demo.mp4": "lot_scene"}

STAGES = [
    ("read", "读取视频"),
    ("detect", "YOLO 检测"),
    ("track", "IoU 跟踪"),
    ("judge", "区域/停留判定"),
    ("write", "告警取证"),
]

app = FastAPI(title="非机动车违停智能检测系统 · 演示")


# ---------------- 任务状态 ----------------
@dataclass
class Job:
    id: str
    source_name: str
    source_path: str
    params: dict
    status: str = "queued"          # queued / running / done / failed / cancelled
    stage: str = "read"
    progress: float = 0.0
    frame_idx: int = 0
    total_frames: int = 0
    fps: float = 0.0
    elapsed: float = 0.0
    detections: int = 0
    intrusion: int = 0
    violations: int = 0
    preview_version: int = 0
    alarms: list = field(default_factory=list)
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    stop: bool = False

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("stop", None)
        d["preview_url"] = (
            f"/media/{self.id}/preview.jpg?v={self.preview_version}"
            if self.preview_version else None
        )
        d["video_url"] = f"/media/{self.id}/annotated.mp4" if self.status == "done" else None
        d["csv_url"] = f"/media/{self.id}/alarms.csv" if self.status == "done" else None
        d["stages"] = [{"key": k, "name": n} for k, n in STAGES]
        return d


JOBS: dict[str, Job] = {}
LOCK = threading.Lock()


def _sample_videos() -> list[dict]:
    out = []
    smp = os.path.join(CODE, "samples")
    if os.path.isdir(smp):
        for fn in sorted(os.listdir(smp)):
            if not fn.lower().endswith((".mp4", ".avi", ".mov")):
                continue
            path = os.path.join(smp, fn)
            cap = cv2.VideoCapture(path)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            zone_pts = None
            meta_path = os.path.join(smp, os.path.splitext(fn)[0] + "_meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, encoding="utf-8") as mf:
                        zone_pts = json.load(mf).get("roi")
                except Exception:
                    zone_pts = None
            out.append({
                "id": fn,
                "roi": zone_pts,
                "name": fn,
                "path": path,
                "frames": frames,
                "seconds": round(frames / fps, 1) if fps else 0,
                "size": f"{w}×{h}",
            })
    return out


def _run_job(job: Job):
    """在后台线程里跑真实流水线。"""
    try:
        cfg = load_config(os.path.join(CODE, "config.yaml"))
        cfg._data["model"]["conf"] = float(job.params["conf"])
        pts = job.params.get("roi_points")
        if pts and len(pts) >= 3:
            roi_file = os.path.join(CODE, "configs", "roi.json")
            with open(roi_file, encoding="utf-8") as f:
                rj = json.load(f)
            rj.setdefault("zones", {})["lot_custom"] = pts
            with open(roi_file, "w", encoding="utf-8") as f:
                json.dump(rj, f, ensure_ascii=False, indent=2)
            cfg._data["roi"]["active"] = "lot_custom"
        else:
            zone_name = SAMPLE_ROI.get(os.path.basename(job.source_path))
            if zone_name:
                cfg._data["roi"]["active"] = zone_name
        if job.params.get("weights"):
            cfg._data["model"]["weights"] = job.params["weights"]
        if job.params.get("mock"):
            cfg._data["model"]["mock"] = True
        cfg._data["judge"]["dwell_seconds"] = float(job.params["dwell"])
        cfg._data["judge"]["mode"] = str(job.params["mode"])
        cfg._data["output"]["dir"] = os.path.join(RUNS, job.id)
        cfg._data["output"]["save_csv"] = True
        cfg._data["output"]["save_screenshot"] = True
        cfg._data["output"]["csv_name"] = "alarms.csv"
        cfg._data["output"]["show_window"] = False

        pipe = Pipeline(cfg)
        job.status = "running"
        job.started_at = time.time()

        last_preview = {"t": 0.0, "idx": -1}

        def on_frame(idx, total, frame, res, counters):
            now = time.time()
            job.frame_idx = idx
            job.intrusion = counters["intrusion"]
            job.violations = counters["violations"]
            job.progress = (idx / total) if total else 0.0
            job.elapsed = now - job.started_at

            # 阶段推断（用于前端点亮 pipeline）
            if idx < 3:
                job.stage = "read"
            elif res is not None and len(res.events) > 0:
                job.stage = "write"
            elif counters["intrusion"] or counters["violations"]:
                job.stage = "judge"
            else:
                job.stage = "detect"

            # 每个事件都记录（带截图路径）
            for ev in res.events:
                job.alarms.append({
                    "kind": ev.kind,
                    "ts": round(float(ev.ts), 2),
                    "tid": ev.tid,
                    "cls": pipe.detector.names.get(ev.cls_id, str(ev.cls_id)),
                    "conf": round(float(ev.conf), 3),
                    "dwell": round(float(ev.dwell), 2),
                    "zone": ev.zone,
                })

            # 预览帧限流（约每 0.4 秒一张）
            if now - last_preview["t"] > 0.4 and idx != last_preview["idx"]:
                last_preview["t"] = now
                last_preview["idx"] = idx
                path = os.path.join(RUNS, job.id, "preview.jpg")
                if _imwrite_unicode(path, frame, quality=80):
                    job.preview_version += 1

        stats = pipe.run(
            source=job.source_path,
            max_frames=0,
            show=False,
            save_video=True,
            on_frame=on_frame,
            stop_flag=lambda: job.stop,
        )
        job.detections = max(job.detections, stats.get("intrusion", 0) + stats.get("violations", 0))
        job.total_frames = stats.get("frames", 0)
        job.progress = 1.0
        job.stage = "write"
        job.status = "cancelled" if job.stop else "done"

        # CSV 里的告警与内存记录对齐：以 CSV 为准补充截图名
        csv_path = os.path.join(RUNS, job.id, "alarms.csv")
        if os.path.exists(csv_path):
            import csv as _csv
            with open(csv_path, encoding="utf-8-sig") as f:
                rows = list(_csv.DictReader(f))
            shots = {r["截图文件"] for r in rows if r.get("截图文件")}
            for i, a in enumerate(job.alarms):
                pass
            job.alarm_images = sorted(shots)
        else:
            job.alarm_images = []
    except Exception as exc:                       # noqa: BLE001
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
    finally:
        job.finished_at = time.time()


# ---------------- API ----------------
class RunRequest(BaseModel):
    source: str                 # 示例文件名 或 upload:<id>
    conf: float = 0.35
    dwell: float = 8.0
    mode: str = "both"
    weights: str = "weights/yolov8n.pt"
    mock: bool = False
    roi_points: list | None = None      # [[x,y],...] 归一化坐标，前端所画


@app.get("/api/state")
def api_state():
    cfg = load_config(os.path.join(CODE, "config.yaml"))
    return {
        "samples": _sample_videos(),
        "defaults": {
            "conf": float(cfg.model.conf),
            "dwell": float(cfg.judge.dwell_seconds),
            "mode": str(cfg.judge.mode),
            "weights": str(cfg.model.weights),
        },
        "weights_options": [
            {"value": "weights/yolov8n.pt", "label": "COCO 预训练（3 类，推荐）"},
            {"value": "weights/ebike_scene_finetuned.pt", "label": "域对齐微调（中文场景，仅摩托车）"},
            {"value": "mock", "label": "演示模式（合成画面色块，无需模型）"},
        ],
    }


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    uid = uuid.uuid4().hex[:8]
    ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    path = os.path.join(UPLOADS, f"{uid}{ext}")
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    if frames == 0:
        os.remove(path)
        raise HTTPException(400, "无法读取视频（格式不支持或文件损坏）")
    return {"upload_id": uid, "name": file.filename, "frames": frames,
            "seconds": round(frames / fps, 1)}


@app.post("/api/run")
def api_run(req: RunRequest):
    if req.source.startswith("upload:"):
        uid = req.source.split(":", 1)[1]
        candidates = [p for p in os.listdir(UPLOADS) if p.startswith(uid)]
        if not candidates:
            raise HTTPException(404, "上传的视频不存在")
        path = os.path.join(UPLOADS, candidates[0])
        name = candidates[0]
    else:
        path = os.path.join(CODE, "samples", req.source)
        name = req.source
    if not os.path.exists(path):
        raise HTTPException(404, f"找不到视频: {req.source}")

    job_id = uuid.uuid4().hex[:10]
    os.makedirs(os.path.join(RUNS, job_id), exist_ok=True)
    job = Job(id=job_id, source_name=name, source_path=path,
              params={"conf": req.conf, "dwell": req.dwell, "mode": req.mode,
                      "weights": req.weights, "mock": req.mock,
                      "roi_points": req.roi_points})
    with LOCK:
        JOBS[job_id] = job
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/frame")
def api_frame(source: str, t: float = 0.0):
    """返回视频某一时刻的帧（JPEG），供前端画禁停区。"""
    from fastapi.responses import Response

    if source.startswith("upload:"):
        uid = source.split(":", 1)[1]
        cands = [p for p in os.listdir(UPLOADS) if p.startswith(uid)]
        if not cands:
            raise HTTPException(404, "上传的视频不存在")
        path = os.path.join(UPLOADS, cands[0])
    else:
        path = os.path.join(CODE, "samples", source)
    if not os.path.exists(path):
        raise HTTPException(404, "找不到视频")

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if t > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(400, "取帧失败")
    ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok2:
        raise HTTPException(500, "编码失败")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/job/{job_id}")
def api_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    data = job.public()
    data["alarm_images"] = getattr(job, "alarm_images", [])
    return data


@app.post("/api/job/{job_id}/cancel")
def api_cancel(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    job.stop = True
    return {"ok": True}


@app.get("/api/jobs")
def api_jobs():
    return [{"id": j.id, "name": j.source_name, "status": j.status,
             "intrusion": j.intrusion, "violations": j.violations,
             "elapsed": round(j.elapsed, 1)} for j in JOBS.values()]


@app.get("/media/{job_id}/{filename}")
def media(job_id: str, filename: str):
    base = os.path.join(RUNS, job_id)
    if ".." in filename or "/" in filename:
        raise HTTPException(400, "非法路径")
    if filename == "preview.jpg":
        path = os.path.join(base, "preview.jpg")
    elif filename.startswith("shot"):
        path = os.path.join(base, "screenshots", filename[2:])
    else:
        path = os.path.join(base, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "文件不存在")
    return FileResponse(path)


@app.get("/shot/{job_id}/{filename}")
def shot(job_id: str, filename: str):
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(400, '非法路径')
    path = os.path.join(RUNS, job_id, 'screenshots', filename)
    if not os.path.isfile(path):
        raise HTTPException(404, '截图不存在')
    return FileResponse(path, media_type='image/jpeg')


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8123, log_level="warning")
