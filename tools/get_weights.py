"""下载 YOLOv8n 预训练权重到 weights/ 目录。

用法
----
python tools/get_weights.py

如果自动下载失败（国内访问 GitHub Release 可能受限），
可以手动下载后放到 weights/yolov8n.pt：
    https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
"""
from __future__ import annotations

import os
import sys
import urllib.request

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(CODE, "weights", "yolov8n.pt")

MIRRORS = [
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "https://ghproxy.net/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
]


def try_urllib(url: str) -> bool:
    print(f"  尝试: {url[:70]}…")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(DST, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total:
                    pct = got * 100 // total
                    print(f"\r    {pct:3d}%  {got/1048576:.1f} MB", end="")
            print()
        return os.path.getsize(DST) > 1024 * 1024
    except Exception as exc:
        print(f"    失败: {type(exc).__name__}")
        if os.path.exists(DST) and os.path.getsize(DST) < 1024 * 1024:
            os.remove(DST)
        return False


def main():
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    if os.path.exists(DST) and os.path.getsize(DST) > 1024 * 1024:
        print(f"权重已存在: {DST}  ({os.path.getsize(DST)/1048576:.1f} MB)")
        return

    print("正在下载 yolov8n.pt（约 6 MB）…")
    for url in MIRRORS:
        if try_urllib(url):
            print(f"完成: {DST}  ({os.path.getsize(DST)/1048576:.1f} MB)")
            return

    print("\n全部镜像失败。请手动下载后放到: weights/yolov8n.pt")
    print("  下载地址: https://gh-proxy.com/https://github.com/ultralytics/assets/"
          "releases/download/v8.3.0/yolov8n.pt")
    sys.exit(1)


if __name__ == "__main__":
    main()
