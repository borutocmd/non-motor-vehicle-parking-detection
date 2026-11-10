"""合成测试视频：用于无权重时验证「检测→跟踪→禁停区→停留→告警→取证」全链路。

画面里两个色块代表车辆（MockDetector 按颜色识别）：
  橙色  → bicycle   驶入禁停区后停住 12 秒（> 阈值 8s）→ 应触发违停
  蓝色  → motorcycle 从画面上方快速横穿 → 不应触发
"""
from __future__ import annotations

import os

import cv2
import numpy as np

W, H, FPS, DUR = 1280, 720, 25, 20
ORANGE = (0, 140, 255)   # bicycle
BLUE = (255, 120, 0)     # motorcycle
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples", "parking_demo.mp4")


def background():
    """浅灰水泥地 + 噪点 + 停车位标线，避免纯色导致检测过于理想化。"""
    bg = np.full((H, W, 3), 168, np.uint8)
    noise = np.random.default_rng(0).normal(0, 6, (H, W, 1)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # 地面透视线
    for i in range(1, 6):
        y = int(H * (0.55 + i * 0.07))
        cv2.line(bg, (0, y), (W, y), (150, 150, 150), 1)
    # 车位分隔线
    for x in range(120, W, 240):
        cv2.line(bg, (x, int(H * 0.55)), (x - 60, H), (200, 200, 200), 3)
    return bg


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(OUT, fourcc, FPS, (W, H))
    bg = background()
    rng = np.random.default_rng(1)

    total = FPS * DUR
    for i in range(total):
        t = i / FPS
        frame = bg.copy()

        # ---- 橙色 bicycle：驶入 → 停 12s → 驶出 ----
        if t < 2.0:
            x = int(-160 + (t / 2.0) * (560 + 160))
            y = int(470 + 20 * np.sin(t * 2))
        elif t < 14.0:
            x, y = 560, 480
        elif t < 16.0:
            p = (t - 14.0) / 2.0
            x = int(560 + p * 900)
            y = 480
        else:
            x, y = -999, -999
        if x > -200 and x < W + 200:
            cv2.rectangle(frame, (x, y), (x + 120, y + 90), ORANGE, -1)
            cv2.rectangle(frame, (x, y), (x + 120, y + 90), (0, 90, 170), 2)
            cv2.circle(frame, (x + 30, y + 92), 13, (30, 30, 30), -1)
            cv2.circle(frame, (x + 90, y + 92), 13, (30, 30, 30), -1)

        # ---- 蓝色 motorcycle：上方快速横穿 ----
        if 3.0 <= t < 6.0:
            p = (t - 3.0) / 3.0
            bx = int(-180 + p * (W + 360))
            by = 150
            cv2.rectangle(frame, (bx, by), (bx + 130, by + 70), BLUE, -1)
            cv2.rectangle(frame, (bx, by), (bx + 130, by + 70), (170, 70, 0), 2)

        # 左上角摄像头信息（模拟真实监控画面）
        cv2.putText(frame, f"CAM-01  {t:05.2f}s", (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

        vw.write(frame)

    vw.release()
    print(f"已生成 {OUT}  {W}x{H} {FPS}fps {DUR}s ({total} 帧)")


if __name__ == "__main__":
    main()
