"""Прогон детектора по всем роликам → JSON для подбора правил.

py -3.11 service/tools/dump_detections.py <папка с .mov> <папка для json>
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analyzer import read_frames  # noqa: E402
from detector import detect  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
vdir, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for name in sorted(os.listdir(vdir)):
    if not name.lower().endswith((".mov", ".mp4")):
        continue
    key = os.path.splitext(name)[0].replace("Х", "X").split("_")[0]
    dst = os.path.join(out, key + ".json")
    if os.path.exists(dst):
        continue
    t0 = time.time()
    frames = read_frames(os.path.join(vdir, name))
    rows = [{"t": round(f.t, 2), "det": detect(f.img)} for f in frames]
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    print(f"{key}: {len(frames)} кадров, {time.time() - t0:.0f} с", flush=True)
