"""Оценка правил на обучающих роликах по закэшированным детекциям.

py -3.11 service/tools/evaluate.py <папка .mov> <open_labels.xlsx> <папка det-json>
Метрики: вердикт по видео (TP/FN/FP/TN) и ошибка таймкода для найденных нарушений.
"""
import json
import os
import re
import sys

import openpyxl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analyzer import read_frames  # noqa: E402
from engine import find_events  # noqa: E402
from rules_extra import calibrate, find_extra  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
vdir, lab, ddir = sys.argv[1:4]
DUP = {"X-47": "X-4", "X-48": "X-5"}          # одинаковые видео с разными метками

truth = {}
for name, ts, _ in list(openpyxl.load_workbook(lab).active.iter_rows(values_only=True))[1:]:
    k = name.replace("Х", "X")
    t = None
    if ts:
        m = re.findall(r"\d+", ts)
        t = int(m[0]) * 60 + int(m[1]) if "мин" in ts else int(m[0])
    k = DUP.get(k, k)
    truth.setdefault(k, [])
    if t is not None:
        truth[k].append(t)

files = {os.path.splitext(f)[0].replace("Х", "X").split("_")[0]: f for f in os.listdir(vdir)}
tp = fn = fp = tn = 0
errs = []
for k in sorted(truth, key=lambda s: int(s.split("-")[1])):
    dets = [r["det"] for r in json.load(open(os.path.join(ddir, k + ".json"), encoding="utf-8"))]
    frames = read_frames(os.path.join(vdir, files[k]))
    ev = calibrate(find_events(frames, dets) + (find_extra(frames, dets) if os.environ.get("EXTRA", "1") == "1" else []), frames)
    pos, pred = bool(truth[k]), bool(ev)
    tp += pos and pred; fn += pos and not pred; fp += pred and not pos; tn += not pos and not pred
    mark = {(1, 1): "TP", (1, 0): "FN", (0, 1): "FP", (0, 0): "TN"}[(pos, pred)]
    desc = ", ".join(f"{e.type}@{e.t:.0f}" for e in ev)
    if pos and pred:
        errs.append(min(abs(e.t - t) for e in ev for t in truth[k]))
    print(f"{k:5} {mark} truth={truth[k]} pred=[{desc}]")
print(f"\nTP={tp} FN={fn} FP={fp} TN={tn}  точность вердикта={(tp + tn) / (tp + fn + fp + tn):.2f}")
if errs:
    print(f"ошибка таймкода: медиана {sorted(errs)[len(errs) // 2]:.1f} с, <=2 с: {sum(e <= 2 for e in errs)}/{len(errs)}")
