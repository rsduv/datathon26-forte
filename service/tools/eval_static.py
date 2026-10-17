"""Проверка пропуска детектора на неподвижных кадрах: подставляем кэш-детекции прошлого кадра, как analyzer.
py -3.11 service/tools/eval_static.py <папка .mov> <open_labels.xlsx> <папка det-json> [пороги через запятую]"""
import json, os, re, sys
import cv2, numpy as np, openpyxl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analyzer import read_frames  # noqa: E402
from engine import find_events  # noqa: E402
sys.stdout.reconfigure(encoding="utf-8")
vdir, lab, ddir = sys.argv[1:4]
ths = [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,1,1.5,2,3").split(",")]
DUP = {"X-47": "X-4", "X-48": "X-5"}
truth = {}
for name, ts, *_ in list(openpyxl.load_workbook(lab).active.iter_rows(values_only=True))[1:]:
    k = DUP.get(name.replace("Х", "X"), name.replace("Х", "X"))
    truth.setdefault(k, [])
    if ts:
        m = re.findall(r"\d+", ts)
        truth[k].append(int(m[0]) * 60 + int(m[1]) if "мин" in ts else int(m[0]))
files = {os.path.splitext(f)[0].replace("Х", "X").split("_")[0]: f for f in os.listdir(vdir)}
cache = {}
for k in truth:
    fr = read_frames(os.path.join(vdir, files[k]))
    dets = [r["det"] for r in json.load(open(os.path.join(ddir, k + ".json"), encoding="utf-8"))]
    sm = [cv2.cvtColor(cv2.resize(f.img, (96, 160)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in fr]
    cache[k] = (fr, dets, sm)
for th in ths:
    tp = fn = fp = tn = calls = total = 0
    for k, (fr, dets, sm) in cache.items():
        out, prev = [], None
        for i in range(min(len(fr), len(dets))):
            if prev is not None and out and float(np.abs(sm[i] - prev).mean()) < th:
                out.append(out[-1])
            else:
                out.append(dets[i]); prev = sm[i]; calls += 1
            total += 1
        pred, pos = bool(find_events(fr, out)), bool(truth[k])
        tp += pos and pred; fn += pos and not pred; fp += pred and not pos; tn += not pos and not pred
    print(f"порог={th}: вызовов детектора {calls}/{total} ({calls / total:.0%})  TP={tp} FN={fn} FP={fp} TN={tn}", flush=True)
