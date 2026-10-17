"""Вклад каждого правила: (найдено - ложные) при добавлении к остальным; на 40 роликах по кэшу детекций.

py -3.11 service/tools/rule_gain.py <папка видео> <open_labels.xlsx> <папка det-json>
"""
import json, os, re, sys
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
    truth.setdefault(k, False)
    truth[k] = truth[k] or bool(ts)
files = {os.path.splitext(f)[0].replace("Х", "X").split("_")[0]: f for f in os.listdir(vdir)}
ALL = [engine.rule_outsider, engine.rule_computer_unlocked, engine.rule_bag_left, engine.rule_door_open_auto,
       engine.rule_quick_exit, engine.rule_short_visit, engine.rule_personal_items]
ACTIVE = set(engine.RULES_FN)
per = {}   # video -> {rule: events}
for k in truth:
    dets = [r["det"] for r in json.load(open(os.path.join(ddir, k + ".json"), encoding="utf-8"))]
    fr = read_frames(os.path.join(vdir, files[k]))
    n = min(len(fr), len(dets))
    fr, dets = engine.normalize_brightness(fr[:n]), dets[:n]
    per[k] = {fn: fn(fr, dets) for fn in ALL}


def score(rules):
    tp = fp = 0
    for k, pos in truth.items():
        ev = engine.dedupe([e for fn in rules for e in per[k][fn]])
        tp += pos and bool(ev)
        fp += (not pos) and bool(ev)
    return tp, fp


base = [fn for fn in ALL if fn in ACTIVE]
tp0, fp0 = score(base)
print(f"текущий набор: найдено {tp0}/{sum(truth.values())}, ложных {fp0}/{len(truth) - sum(truth.values())}")
for fn in ALL:
    if fn in ACTIVE:
        tp, fp = score([f for f in base if f is not fn])
        d_tp, d_fp = tp0 - tp, fp0 - fp
    else:
        tp, fp = score(base + [fn])
        d_tp, d_fp = tp - tp0, fp - fp0
    print(f"{fn.__name__:24} {'вкл ' if fn in ACTIVE else 'выкл'}  +найдено {d_tp}  +ложных {d_fp}  выигрыш {d_tp - d_fp:+d}")
