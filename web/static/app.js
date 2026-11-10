/* 非机动车违停检测 · 前端逻辑 */
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const STAGES = [
  ["read", "读取视频"],
  ["detect", "YOLO 检测"],
  ["track", "IoU 跟踪"],
  ["judge", "区域/停留判定"],
  ["write", "告警取证"],
];

let state = { samples: [], defaults: {}, weights_options: [] };
let selected = null;
let polling = null;
let jobId = null;
let roiPoints = [];          // 归一化坐标 [[x,y],...]
let roiDefault = [];         // 该视频的默认区域
let frameLoadedFor = null;   // 底图对应的 source

/* ---------- 初始化 ---------- */
async function init() {
  renderPipeline();
  const res = await fetch("/api/state");
  state = await res.json();

  const box = $("#samples");
  box.innerHTML = "";
  state.samples.forEach((s, i) => {
    const el = document.createElement("div");
    el.className = "sample" + (i === 0 ? " sel" : "");
    el.innerHTML = `<div class="name">${s.name}</div>
      <div class="meta">${s.frames} 帧 · ${s.seconds}s · ${s.size}</div>`;
    el.onclick = () => {
      $$(".sample").forEach((x) => x.classList.remove("sel"));
      el.classList.add("sel");
      selectSource({ kind: "sample", value: s.name, name: s.name, roi: s.roi });
      gotoStep(2);
    };
    box.appendChild(el);
    if (i === 0) selectSource({ kind: "sample", value: s.name, name: s.name, roi: s.roi });
  });

  $("#conf").value = state.defaults.conf;
  $("#dwell").value = state.defaults.dwell;
  $("#mode").value = state.defaults.mode;
  $("#conf-val").textContent = Number(state.defaults.conf).toFixed(2);
  $("#dwell-val").textContent = state.defaults.dwell;

  const wsel = $("#weights");
  wsel.innerHTML = "";
  state.weights_options.forEach((o) => {
    const opt = document.createElement("option");
    opt.value = o.value; opt.textContent = o.label;
    if (o.value === state.defaults.weights) opt.selected = true;
    wsel.appendChild(opt);
  });

  bindEvents();
  $("#engine-badge").textContent = "就绪 · " + state.samples.length + " 个示例";
}

/* ---------- 选择视频源 ---------- */
function selectSource(src) {
  selected = src;
  roiDefault = (src.roi || []).map((p) => [p[0], p[1]]);
  roiPoints = roiDefault.map((p) => [p[0], p[1]]);
  autoModel(src.name);
  loadFrame();
}

async function loadFrame() {
  if (!selected) return;
  const img = $("#roi-bg");
  const url = `/api/frame?source=${encodeURIComponent(selected.value)}&t=1`;
  frameLoadedFor = selected.value;
  img.onload = () => {
    const cv = $("#roi-canvas");
    cv.width = img.naturalWidth;
    cv.height = img.naturalHeight;
    $("#roi-empty").style.display = "none";
    drawRoi();
    updateRoiCount();
  };
  img.onerror = () => { $("#roi-empty").style.display = "block"; };
  img.src = url;
}

/* ---------- ROI 绘制 ---------- */
function canvasPoint(ev) {
  const cv = $("#roi-canvas");
  const r = cv.getBoundingClientRect();
  return [
    Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)),
    Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height)),
  ];
}

function drawRoi() {
  const cv = $("#roi-canvas");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (roiPoints.length === 0) return;

  const pts = roiPoints.map(([x, y]) => [x * cv.width, y * cv.height]);
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  pts.slice(1).forEach((p) => ctx.lineTo(p[0], p[1]));
  if (pts.length >= 3) {
    ctx.closePath();
    ctx.fillStyle = "rgba(0, 215, 255, 0.20)";
    ctx.fill();
  }
  ctx.strokeStyle = "rgba(0, 215, 255, 0.95)";
  ctx.lineWidth = Math.max(2, cv.width / 420);
  ctx.stroke();

  pts.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p[0], p[1], Math.max(4, cv.width / 260), 0, Math.PI * 2);
    ctx.fillStyle = i === 0 ? "#ffb703" : "#00d7ff";
    ctx.fill();
    ctx.strokeStyle = "#06111f";
    ctx.lineWidth = 2;
    ctx.stroke();
  });
}

function updateRoiCount() {
  const n = roiPoints.length;
  $("#roi-count").textContent = n ? `${n} 个顶点（${n >= 3 ? "可用" : "至少 3 个"}）` : "未设置（用默认区域）";
}

/* ---------- 事件 ---------- */
function bindEvents() {
  const cv = $("#roi-canvas");
  cv.onclick = (ev) => {
    if (frameLoadedFor !== (selected && selected.value)) return;
    roiPoints.push(canvasPoint(ev));
    drawRoi();
    updateRoiCount();
  };
  cv.oncontextmenu = (ev) => {
    ev.preventDefault();
    roiPoints.pop();
    drawRoi();
    updateRoiCount();
  };

  $("#roi-undo").onclick = () => { roiPoints.pop(); drawRoi(); updateRoiCount(); };
  $("#roi-clear").onclick = () => { roiPoints = []; drawRoi(); updateRoiCount(); };
  $("#roi-default").onclick = () => {
    roiPoints = roiDefault.map((p) => [p[0], p[1]]);
    drawRoi(); updateRoiCount();
  };
  $("#roi-full").onclick = () => {
    roiPoints = [[0.02, 0.02], [0.98, 0.02], [0.98, 0.98], [0.02, 0.98]];
    drawRoi(); updateRoiCount();
  };

  $("#conf").oninput = (e) => { $("#conf-val").textContent = Number(e.target.value).toFixed(2); };
  $("#dwell").oninput = (e) => { $("#dwell-val").textContent = e.target.value; };

  $("#btn-random").onclick = () => {
    const conf = (0.2 + Math.random() * 0.35).toFixed(2);
    const dwell = Math.round(4 + Math.random() * 8);
    $("#conf").value = conf; $("#conf-val").textContent = conf;
    $("#dwell").value = dwell; $("#dwell-val").textContent = dwell;
    $("#mode").value = Math.random() > 0.5 ? "both" : "dwell";
    flash($("#btn-random"), "已随机");
  };

  $("#file-input").onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    $("#upload-info").textContent = "上传中…";
    const fd = new FormData();
    fd.append("file", f);
    try {
      const r = await fetch("/api/upload", { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.json()).detail || "上传失败");
      const d = await r.json();
      $$(".sample").forEach((x) => x.classList.remove("sel"));
      selectSource({ kind: "upload", value: "upload:" + d.upload_id, name: d.name, roi: null });
      $("#upload-info").textContent = `已选：${d.name}（${d.frames} 帧 / ${d.seconds}s）`;
      gotoStep(2);
    } catch (err) {
      $("#upload-info").textContent = "上传失败：" + err.message;
    }
  };

  $("#btn-run").onclick = runJob;
  $("#btn-cancel").onclick = async () => {
    if (!jobId) return;
    await fetch(`/api/job/${jobId}/cancel`, { method: "POST" });
  };

  $("#lightbox").onclick = () => { $("#lightbox").hidden = true; };
}

function autoModel(name) {
  const sel = $("#weights");
  if (!sel) return;
  if (/^parking_demo/i.test(name)) sel.value = "mock";
  else if (sel.value === "mock") sel.value = "weights/yolov8n.pt";
}

/* ---------- 运行 ---------- */
async function runJob() {
  if (!selected) { alert("请先选择视频"); return; }
  resetResult();

  $("#btn-run").disabled = true;
  $("#btn-cancel").disabled = false;
  $("#engine-badge").textContent = "运行中…";
  $("#engine-badge").classList.add("on");
  gotoStep(3);

  const body = {
    source: selected.value,
    conf: Number($("#conf").value),
    dwell: Number($("#dwell").value),
    mode: $("#mode").value,
    weights: $("#weights").value === "mock" ? "weights/yolov8n.pt" : $("#weights").value,
    mock: $("#weights").value === "mock",
    roi_points: roiPoints.length >= 3
      ? roiPoints.map(([x, y]) => [+x.toFixed(4), +y.toFixed(4)])
      : null,
  };

  try {
    const r = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error((await r.json()).detail || "启动失败");
    const d = await r.json();
    jobId = d.job_id;
    poll();
  } catch (err) {
    $("#progress-text").textContent = "启动失败：" + err.message;
    finishRun();
  }
}

function poll() {
  clearInterval(polling);
  polling = setInterval(async () => {
    if (!jobId) return;
    let d;
    try {
      d = await (await fetch(`/api/job/${jobId}`)).json();
    } catch { return; }

    updateProgress(d);
    if (d.status === "done" || d.status === "failed" || d.status === "cancelled") {
      clearInterval(polling);
      polling = null;
      finishRun();
      showResult(d);
    }
  }, 450);
}

function updateProgress(d) {
  const pct = Math.max(0, Math.min(1, d.progress || 0));
  $("#bar").style.width = (pct * 100).toFixed(1) + "%";

  const stageName = STAGES.find((s) => s[0] === d.stage);
  $("#progress-text").textContent =
    d.status === "queued" ? "排队中…" :
    d.status === "running" ? `处理中 · ${stageName ? stageName[1] : ""} · 第 ${d.frame_idx} 帧` :
    d.status === "done" ? "已完成" :
    d.status === "cancelled" ? "已中止" : "失败：" + (d.error || "");

  $("#progress-time").textContent = `耗时 ${(d.elapsed || 0).toFixed(1)}s`;
  $("#stat-frames").textContent = d.frame_idx || 0;
  $("#stat-intrusion").textContent = d.intrusion || 0;
  $("#stat-violation").textContent = d.violations || 0;
  $("#stat-time").textContent = (d.elapsed || 0).toFixed(1) + "s";

  const order = STAGES.map((s) => s[0]);
  const cur = order.indexOf(d.stage);
  $$(".stage").forEach((el, i) => {
    el.classList.toggle("on", i === cur && d.status === "running");
    el.classList.toggle("ok", i < cur || d.status === "done");
  });

  if (d.preview_url) {
    const img = $("#preview");
    img.src = d.preview_url;
    img.hidden = false;
    $("#live-empty").style.display = "none";
  }
}

function finishRun() {
  $("#btn-run").disabled = false;
  $("#btn-cancel").disabled = true;
  $("#engine-badge").classList.remove("on");
  $("#engine-badge").textContent = "就绪";
}

/* ---------- 结果 ---------- */
function showResult(d) {
  gotoStep(4);
  const zone = d.params.roi_points ? "自定义多边形" : "默认区域";
  $("#result-hint").textContent =
    `${d.source_name} · conf=${d.params.conf} · 停留=${d.params.dwell}s · 模式=${d.params.mode} · ${zone}`;

  $("#stat-frames").textContent = d.total_frames || d.frame_idx || 0;
  $("#stat-intrusion").textContent = d.intrusion || 0;
  $("#stat-violation").textContent = d.violations || 0;
  $("#stat-time").textContent = (d.elapsed || 0).toFixed(1) + "s";

  const dl = $("#downloads");
  dl.innerHTML = "";
  if (d.video_url) dl.innerHTML += `<a href="${d.video_url}" target="_blank">下载标注视频</a>`;
  if (d.csv_url) dl.innerHTML += `<a href="${d.csv_url}" target="_blank">下载告警 CSV</a>`;

  const tl = $("#timeline");
  tl.innerHTML = "";
  if (!d.alarms || d.alarms.length === 0) {
    tl.innerHTML = '<p class="empty">本次未触发告警（车辆未进入禁停区或未达阈值）</p>';
  } else {
    d.alarms.forEach((a) => {
      const el = document.createElement("div");
      el.className = "tl-item " + a.kind;
      el.innerHTML = `<span class="t">${a.ts}s</span>
        <span class="k">${a.kind === "intrusion" ? "区域入侵" : "违停"}</span>
        <span class="d">${a.cls} #${a.tid} · 置信度 ${a.conf} · 停留 ${a.dwell}s · 区域 ${a.zone}</span>`;
      tl.appendChild(el);
    });
  }

  const sh = $("#shots");
  sh.innerHTML = "";
  const imgs = d.alarm_images || [];
  if (imgs.length === 0) {
    sh.innerHTML = '<p class="empty">暂无截图</p>';
  } else {
    imgs.forEach((name) => {
      const url = `/shot/${d.id}/${encodeURIComponent(name)}`;
      const img = document.createElement("img");
      img.src = url + "?v=" + Date.now();
      img.alt = name;
      img.onclick = () => {
        $("#lightbox-img").src = url;
        $("#lightbox").hidden = false;
      };
      sh.appendChild(img);
    });
  }

  if (d.video_url) {
    $("#video-wrap").hidden = false;
    $("#result-video").src = d.video_url;
  }
}

function resetResult() {
  $("#timeline").innerHTML = '<p class="empty">暂无告警</p>';
  $("#shots").innerHTML = '<p class="empty">暂无截图</p>';
  $("#downloads").innerHTML = "";
  $("#video-wrap").hidden = true;
  ["stat-frames", "stat-intrusion", "stat-violation"].forEach((id) => { $("#" + id).textContent = "0"; });
  $("#stat-time").textContent = "0s";
  $("#bar").style.width = "0%";
  $$(".stage").forEach((el) => el.classList.remove("on", "ok"));
}

/* ---------- 步骤 ---------- */
function gotoStep(n) {
  $$(".step").forEach((el) => {
    const i = Number(el.dataset.step);
    el.classList.toggle("active", i === n);
    el.classList.toggle("done", i < n);
  });
}

function renderPipeline() {
  const box = $("#pipeline");
  box.innerHTML = "";
  STAGES.forEach(([key, name]) => {
    const el = document.createElement("div");
    el.className = "stage";
    el.dataset.key = key;
    el.textContent = name;
    box.appendChild(el);
  });
}

function flash(el, text) {
  const old = el.textContent;
  el.textContent = text;
  setTimeout(() => { el.textContent = old; }, 900);
}

/* ---------- URL 便捷参数 ---------- */
async function handleUrlParams() {
  const q = new URLSearchParams(location.search);
  const job = q.get("job");
  if (job) {
    try {
      const d = await (await fetch(`/api/job/${job}`)).json();
      if (d && d.id) {
        jobId = d.id;
        showResult(d);
        if (d.preview_url) {
          $("#preview").src = d.preview_url;
          $("#preview").hidden = false;
          $("#live-empty").style.display = "none";
        }
        $("#progress-text").textContent = "已载入历史结果";
        $("#bar").style.width = "100%";
        $$(".stage").forEach((el) => el.classList.add("ok"));
      }
    } catch (e) { /* 忽略 */ }
    return;
  }
  if (q.get("autorun")) {
    setTimeout(() => $("#btn-run").click(), 600);
  }
}

init().then(handleUrlParams);
