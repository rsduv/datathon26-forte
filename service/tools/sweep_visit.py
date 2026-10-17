"""Подбор порога rule_short_visit: найдено/ложные при разных max_sec (по кэшу детекций).

py -3.11 service/tools/sweep_visit.py <папка видео> <open_labels.xlsx> <папка det-json>
"""
import json, os, sys
import openpyxl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import engine  # noqa: E402
from analyzer import read_frames  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
vdir, lab, ddir = sys.argv[1:4]
DUP = {"X-47": "X-4", "X-48": "X-5"}
truth = {}
for name, ts, *_ in list(openpyxl.load_workbook(lab).active.iter_rows(values_only=True))[1:]:
    k = DUP.get(name.replace("Х", "X"), name.replace("Х", "X"))
    truth[k] = truth.get(k, False) or bool(ts)
files = {os.path.splitext(f)[0].replace("Х", "X").split("_")[0]: f for f in os.listdir(vdir)}
others = [f for f in engine.RULES_FN if f is not engine.rule_short_visit]
data = {}
for k in truth:
    dets = [r["det"] for r in json.load(open(os.path.join(ddir, k + ".json"), encoding="utf-8"))]
    fr = read_frames(os.path.join(vdir, files[k])); n = min(len(fr), len(dets))
    fr, dets = engine.normalize_brightness(fr[:n]), dets[:n]
    data[k] = (fr, dets, bool(engine.dedupe([e for f in others for e in f(fr, dets)])))
for ms in (10, 11, 12, 13, 14):
    tp = fp = 0; caught = []
    for k, (fr, dets, base) in data.items():
        hit = base or bool(engine.rule_short_visit(fr, dets, max_sec=ms))
        tp += truth[k] and hit; fp += (not truth[k]) and hit
        if hit and not base: caught.append(k + ("" if truth[k] else "(ЛОЖН)"))
    print(f"max_sec={ms}: найдено {tp}/21, ложных {fp}/19, короткий визит ловит: {caught}")
