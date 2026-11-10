"""违停 / 区域入侵判定状态机。

双层告警的设计
--------------------
1. **区域入侵**（必须）：非机动车一进入禁停区就告警、截图。
2. **违停**（进阶）：在区域内持续停留超过阈值 → 升级为"违停"，再告警一次。

这样既满足「进入即触发告警」的硬性要求，又用停留时长把「临时路过」和「真停放」
区分开，避免画面里一有车就报警。

状态
----
    outside    没进区域
    inside     在区域内，尚未告警
    intrusion  已触发入侵告警（进门就报）
    violation  停留超时，升级为违停

配置（config.yaml → judge）
--------------------------
    mode: both          # intrusion / dwell / both
    dwell_seconds: 8.0  # 停留多久升级为违停
    min_hits: 5         # 停留判定需要的连续命中帧数
    intrusion_hits: 2   # 入侵判定需要的连续帧数（防抖）
    cooldown_seconds: 30
    alarm_class_ids: [1, 3]   # 只有非机动车触发告警（汽车只检测、不告警）
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .roi import RoiZone, anchor_of_box
from .tracker import Track

OUTSIDE, INSIDE, INTRUSION, VIOLATION = "outside", "inside", "intrusion", "violation"
KIND_INTRUSION, KIND_VIOLATION = "intrusion", "violation"


@dataclass
class TrackState:
    inside_since: float | None = None
    inside_hits: int = 0
    alerted_intrusion: bool = False     # 本次停留是否已报过入侵
    alerted_violation: bool = False     # 本次停留是否已报过违停
    last_intrusion_at: float | None = None   # 入侵告警的冷却时间戳
    last_violation_at: float | None = None   # 违停告警的冷却时间戳（与入侵互不影响）
    violations: int = 0
    last_status: str = OUTSIDE
    peak_dwell: float = 0.0


@dataclass
class AlarmEvent:
    """一次告警；kind 区分「入侵」与「违停」。"""
    kind: str
    tid: int
    ts: float
    dwell: float
    box: tuple[float, float, float, float]
    cls_id: int
    conf: float
    anchor: tuple[float, float]
    zone: str


ViolationEvent = AlarmEvent  # 兼容旧名字


@dataclass
class JudgeResult:
    states: dict[int, TrackState] = field(default_factory=dict)
    status: dict[int, str] = field(default_factory=dict)
    dwell: dict[int, float] = field(default_factory=dict)
    events: list[AlarmEvent] = field(default_factory=list)

    @property
    def total_violations(self) -> int:
        return sum(s.violations for s in self.states.values())

    def count(self, kind: str) -> int:
        return sum(1 for e in self.events if e.kind == kind)


class ViolationJudge:
    def __init__(self, dwell_seconds: float = 8.0, min_hits: int = 5,
                 cooldown_seconds: float = 30.0, require_inside: bool = True,
                 anchor: str = "bottom_center", mode: str = "both",
                 intrusion_hits: int = 2,
                 alarm_class_ids: list[int] | None = None):
        self.dwell_seconds = float(dwell_seconds)
        self.min_hits = int(min_hits)
        self.cooldown_seconds = float(cooldown_seconds)
        self.require_inside = bool(require_inside)
        self.anchor = anchor
        self.mode = mode if mode in ("intrusion", "dwell", "both") else "both"
        self.intrusion_hits = max(1, int(intrusion_hits))
        self.alarm_class_ids = set(alarm_class_ids) if alarm_class_ids else None
        self.states: dict[int, TrackState] = {}
        self.stats = {KIND_INTRUSION: 0, KIND_VIOLATION: 0}

    def reset(self):
        self.states.clear()
        self.stats = {KIND_INTRUSION: 0, KIND_VIOLATION: 0}

    def _should_alarm(self, cls_id: int) -> bool:
        return self.alarm_class_ids is None or cls_id in self.alarm_class_ids

    def _cooling(self, last: float | None, now_ts: float) -> bool:
        return last is not None and (now_ts - last) < self.cooldown_seconds

    def update(self, tracks: list[Track], zone: RoiZone | None, w: int, h: int,
               ts: float, now: float | None = None) -> JudgeResult:
        """ts 是视频时间；now 是真实时间（视频回放时冷却应按真实时间算）。"""
        res = JudgeResult()
        alive = {t.tid for t in tracks}
        for tid in list(self.states):
            if tid not in alive:
                del self.states[tid]

        wall = ts if now is None else now

        for t in tracks:
            st = self.states.setdefault(t.tid, TrackState())
            anch = anchor_of_box(t.box, self.anchor)
            inside = bool(zone and zone.contains(anch, w, h)) if zone else (not self.require_inside)

            if not inside:
                # 离开即清零：重新进入会再次告警（冷却期外）——「路过」不误报的关键
                st.inside_since = None
                st.inside_hits = 0
                st.alerted_intrusion = False
                st.alerted_violation = False
                st.last_status = OUTSIDE
                res.states[t.tid] = st
                res.status[t.tid] = OUTSIDE
                res.dwell[t.tid] = 0.0
                continue

            if st.inside_since is None:
                st.inside_since = ts
                st.inside_hits = 1
            else:
                st.inside_hits += 1

            dwell = max(0.0, ts - st.inside_since)
            st.peak_dwell = max(st.peak_dwell, dwell)

            # ---- 第一层：区域入侵（进入即告警）----
            if (self.mode in ("intrusion", "both")
                    and st.inside_hits >= self.intrusion_hits
                    and not st.alerted_intrusion
                    and not self._cooling(st.last_intrusion_at, wall)
                    and self._should_alarm(t.cls_id)):
                st.alerted_intrusion = True
                st.last_intrusion_at = wall
                st.last_status = INTRUSION
                self.stats[KIND_INTRUSION] += 1
                res.events.append(AlarmEvent(
                    kind=KIND_INTRUSION, tid=t.tid, ts=ts, dwell=dwell, box=t.box,
                    cls_id=t.cls_id, conf=t.conf, anchor=anch,
                    zone=zone.name if zone else "-"))

            # ---- 第二层：停留超时 → 违停 ----
            if (self.mode in ("dwell", "both")
                    and dwell >= self.dwell_seconds
                    and st.inside_hits >= self.min_hits
                    and not st.alerted_violation
                    and not self._cooling(st.last_violation_at, wall)
                    and self._should_alarm(t.cls_id)):
                st.alerted_violation = True
                st.last_violation_at = wall
                st.violations += 1
                st.last_status = VIOLATION
                self.stats[KIND_VIOLATION] += 1
                res.events.append(AlarmEvent(
                    kind=KIND_VIOLATION, tid=t.tid, ts=ts, dwell=dwell, box=t.box,
                    cls_id=t.cls_id, conf=t.conf, anchor=anch,
                    zone=zone.name if zone else "-"))
            elif st.last_status != VIOLATION:
                st.last_status = INTRUSION if st.alerted_intrusion else INSIDE

            res.states[t.tid] = st
            res.status[t.tid] = st.last_status
            res.dwell[t.tid] = dwell

        return res
