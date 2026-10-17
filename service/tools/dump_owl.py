"""OWLv2-оценки предметов на кропе стола по всем роликам → JSON (для подбора правил по мелким предметам).

py -3.11 service/tools/dump_owl.py <папка .mov> <папка det-json> <вывод.json> [кадров_на_видео]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analyzer import read_frames  # noqa: E402
from engine import table_box  # noqa: E402
from rules_extra import desk_crop  # noqa: E402
from owl import score  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
vdir, ddir, out = sys.argv[1:4]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 8
Q = ["a coffee cup", "a paper cup", "a handbag", "a jacket", "a stack of banknotes"]
res = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else {}
for name in sorted(os.listdir(vdir)):
    key = os.path.splitext(name)[0].replace("Х", "X").split("_")[0]
    if not name.lower().endswith((".mov", ".mp4")) or key in res or key in ("X-47", "X-48"):
        continue
    dets = [r["det"] for r in json.load(open(os.path.join(ddir, key + ".json"), encoding="utf-8"))]
    fr = read_frames(os.path.join(vdir, name)); n = min(len(fr), len(dets))
    box = table_box(dets[:n])
    idx = sorted(set(np.linspace(0, n - 1, K).astype(int).tolist()))
    res[key] = {"q": Q, "t": [round(fr[i].t, 1) for i in idx],
                "s": [[round(float(v), 3) for v in score(desk_crop(fr[i].img, box), Q)] for i in idx]}
    json.dump(res, open(out, "w", encoding="utf-8"))
    print(key, "ok", flush=True)
