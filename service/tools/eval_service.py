"""Сквозная проверка: каждое видео целиком через analyzer.analyze (как в сервисе), сверка с таблицей.
py -3.11 service/tools/eval_service.py <папка .mov> <open_labels.xlsx> [out.json]"""
import json, os, re, sys, time
import openpyxl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analyzer import analyze  # noqa: E402
sys.stdout.reconfigure(encoding="utf-8")
vdir, lab = sys.argv[1:3]
out = sys.argv[3] if len(sys.argv) > 3 else "eval_service.json"
DUP = {"X-47": "X-4", "X-48": "X-5"}
truth = {}
for name, ts, *_ in list(openpyxl.load_workbook(lab).active.iter_rows(values_only=True))[1:]:
    k = DUP.get(name.replace("Х", "X"), name.replace("Х", "X"))
    truth.setdefault(k, [])
    if ts:
        m = re.findall(r"\d+", ts)
        truth[k].append(int(m[0]) * 60 + int(m[1]) if "мин" in ts else int(m[0]))
files = {os.path.splitext(f)[0].replace("Х", "X").split("_")[0]: f for f in os.listdir(vdir)}
res, c = {}, dict(tp=0, fn=0, fp=0, tn=0)
for k in sorted(truth, key=lambda s: int(s.split("-")[1])):
    t0 = time.time()
    try:
        r = analyze(os.path.join(vdir, files[k]))
    except Exception as e:  # noqa: BLE001
        print(k, "ОШИБКА", e, flush=True); res[k] = {"error": str(e)}; continue
    sec = time.time() - t0
    pred, pos = r["verdict"] == "violation", bool(truth[k])
    mark = {(1, 1): "tp", (1, 0): "fn", (0, 1): "fp", (0, 0): "tn"}[(pos, pred)]
    c[mark] += 1
    v = [(x["type"], x["timecode_sec"]) for x in r["violations"]]
    res[k] = {"mark": mark, "sec": round(sec, 1), "truth": truth[k], "pred": v}
    print(f"{k:5} {mark.upper()} {sec:5.0f}s truth={truth[k]} pred={v}", flush=True)
n = sum(c.values())
print(f"\nTP={c['tp']} FN={c['fn']} FP={c['fp']} TN={c['tn']}  точность={(c['tp'] + c['tn']) / n:.2f}  "
      f"макс. время={max(x.get('sec', 0) for x in res.values()):.0f}s", flush=True)
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
