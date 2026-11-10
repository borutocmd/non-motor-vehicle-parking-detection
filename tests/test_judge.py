"""判定逻辑单元测试（不依赖权重/视频，纯逻辑验证）。

覆盖区域入侵与停留判定的 8 个场景：
  1. 进入即告警（区域入侵，必须满足）
  2. 停留够久 → 升级为违停
  3. 停留不足 → 只有入侵告警
  4. 离开后重新进入 → 再告警
  5. 冷却期内不重复告警
  6. 区域外目标永不触发
  7. 汽车只检测不告警
  8. mode=dwell 时退回旧行为
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.judge import (INTRUSION, KIND_INTRUSION, KIND_VIOLATION,
                       OUTSIDE, VIOLATION, ViolationJudge)
from src.roi import RoiZone
from src.tracker import Track

W, H = 1000, 1000
ZONE = RoiZone(name="test", points_norm=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)])
BIKE, MOTO, CAR = 1, 3, 2


def track(tid, cx, cy, box_w=80, box_h=80, cls_id=BIKE):
    """构造一个底边中点落在 (cx, cy) 的轨迹。"""
    x1, y1 = cx - box_w / 2, cy - box_h
    return Track(tid=tid, box=(x1, y1, x1 + box_w, y1 + box_h), cls_id=cls_id, conf=0.9)


def kinds(events):
    return [e.kind for e in events]


class TestIntrusionAndViolation(unittest.TestCase):
    def test_intrusion_alarms_immediately(self):
        """进入区域第 2 帧就该告警（intrusion_hits=2）。"""
        j = ViolationJudge(dwell_seconds=8.0, min_hits=3, intrusion_hits=2, mode="both")
        seen = []
        for i in range(5):
            r = j.update([track(1, 500, 500)], ZONE, W, H, float(i))
            seen += kinds(r.events)
        self.assertIn(KIND_INTRUSION, seen, "进入禁停区后必须立即触发入侵告警")
        self.assertEqual(seen.count(KIND_INTRUSION), 1, "入侵告警每个目标只报一次")

    def test_dwell_upgrades_to_violation(self):
        """停留 15 秒：1 次入侵 + 1 次违停。"""
        j = ViolationJudge(dwell_seconds=8.0, min_hits=3, intrusion_hits=2, mode="both")
        seen = []
        for i in range(16):
            r = j.update([track(1, 500, 500)], ZONE, W, H, float(i))
            seen += kinds(r.events)
        self.assertEqual(seen.count(KIND_INTRUSION), 1)
        self.assertEqual(seen.count(KIND_VIOLATION), 1)
        self.assertEqual(r.status[1], VIOLATION)

    def test_short_stay_only_intrusion(self):
        """停 6 秒（< 8s 阈值）：只有入侵告警，不升级为违停。"""
        j = ViolationJudge(dwell_seconds=8.0, min_hits=3, mode="both")
        seen = []
        for i in range(7):
            r = j.update([track(1, 500, 500)], ZONE, W, H, float(i))
            seen += kinds(r.events)
        self.assertEqual(seen.count(KIND_INTRUSION), 1)
        self.assertEqual(seen.count(KIND_VIOLATION), 0)
        self.assertEqual(r.status[1], INTRUSION)

    def test_leave_resets_and_reentry_alarms_again(self):
        """离开后重新进入：应再次触发入侵告警（冷却期外）。"""
        j = ViolationJudge(dwell_seconds=8.0, min_hits=2, intrusion_hits=2,
                           cooldown_seconds=2.0, mode="both")
        seen = []
        for i in range(4):                       # 进入
            seen += kinds(j.update([track(1, 500, 500)], ZONE, W, H, float(i)).events)
        for i in range(4, 8):                    # 离开
            seen += kinds(j.update([track(1, 500, 100)], ZONE, W, H, float(i)).events)
        for i in range(8, 12):                   # 再进入
            seen += kinds(j.update([track(1, 500, 500)], ZONE, W, H, float(i)).events)
        self.assertEqual(seen.count(KIND_INTRUSION), 2, "两次进入应各告警一次")

    def test_cooldown_blocks_repeat_violation(self):
        """持续停留 40 秒、cooldown=30：违停只报 1 次。"""
        j = ViolationJudge(dwell_seconds=8.0, min_hits=2, cooldown_seconds=30.0, mode="both")
        seen = []
        for i in range(41):
            seen += kinds(j.update([track(1, 500, 500)], ZONE, W, H, float(i)).events)
        self.assertEqual(seen.count(KIND_VIOLATION), 1)

    def test_outside_never_triggers(self):
        j = ViolationJudge(dwell_seconds=1.0, min_hits=1, mode="both")
        for i in range(30):
            r = j.update([track(1, 100, 100)], ZONE, W, H, float(i))
            self.assertEqual(len(r.events), 0)
            self.assertEqual(r.status[1], OUTSIDE)

    def test_car_detected_but_not_alarmed(self):
        """汽车在禁停区里也不告警（只有非机动车才触发）。"""
        j = ViolationJudge(dwell_seconds=1.0, min_hits=1, mode="both",
                           alarm_class_ids=[BIKE, MOTO])
        for i in range(10):
            r = j.update([track(9, 500, 500, cls_id=CAR)], ZONE, W, H, float(i))
            self.assertEqual(len(r.events), 0, "汽车不应触发告警")

        # 同位置换成摩托车 → 应告警
        seen = []
        for i in range(11, 14):
            seen += kinds(j.update([track(10, 500, 500, cls_id=MOTO)], ZONE, W, H, float(i)).events)
        self.assertIn(KIND_INTRUSION, seen, "摩托车进入应告警")

    def test_dwell_only_mode(self):
        """mode=dwell 时不报入侵，只报违停（旧行为）。"""
        j = ViolationJudge(dwell_seconds=5.0, min_hits=2, mode="dwell")
        seen = []
        for i in range(10):
            seen += kinds(j.update([track(1, 500, 500)], ZONE, W, H, float(i)).events)
        self.assertEqual(seen.count(KIND_INTRUSION), 0)
        self.assertEqual(seen.count(KIND_VIOLATION), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
