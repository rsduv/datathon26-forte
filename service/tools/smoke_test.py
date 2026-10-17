"""Проверка API так, как его будет дёргать жюри: ключ, загрузка .mov, опрос статуса, схема ответа, время.

python tools/smoke_test.py <видео.mov> [--url http://localhost:8000] [--key dev-key]
"""
import argparse
import base64
import sys
import time

import requests

p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("--url", default="http://localhost:8000")
p.add_argument("--key", default="dev-key")
a = p.parse_args()
url = a.url.rstrip("/")
fail = []

r = requests.post(f"{url}/api/v1/analyze", files={"file": open(a.video, "rb")})
if r.status_code != 401:
    fail.append(f"без ключа ожидался 401, получен {r.status_code}")

t0 = time.time()
r = requests.post(f"{url}/api/v1/analyze", headers={"X-API-Key": a.key},
                  files={"file": open(a.video, "rb")})
r.raise_for_status()
job_id = r.json()["job_id"]
while True:
    j = requests.get(f"{url}/api/v1/jobs/{job_id}", headers={"X-API-Key": a.key}).json()
    if j["status"] != "processing":
        break
    if time.time() - t0 > 300:
        fail.append("не уложились в 5 минут")
        break
    time.sleep(2)
dt = time.time() - t0

if j.get("status") != "done":
    fail.append(f"статус {j.get('status')}: {j.get('error')}")
else:
    if j.get("verdict") not in ("violation", "no_violation"):
        fail.append(f"неверный verdict: {j.get('verdict')}")
    vs = j.get("violations")
    if not isinstance(vs, list) or (j["verdict"] == "violation") != bool(vs):
        fail.append("verdict не согласован со списком violations")
    for v in vs or []:
        for k, typ in (("type", str), ("timecode_sec", (int, float)), ("description", str), ("evidence_frames", list)):
            if not isinstance(v.get(k), typ):
                fail.append(f"поле {k} отсутствует или неверного типа")
        if not 1 <= len(v.get("evidence_frames", [])) <= 3:
            fail.append("кадров-доказательств должно быть 1–3")
        for f in v.get("evidence_frames", []):
            if not base64.b64decode(f["image_base64"])[:2] == b"\xff\xd8":
                fail.append("кадр не JPEG")
    print(f"{j['verdict']}: " + "; ".join(f"{v['type']} @ {v['timecode_sec']} с" for v in vs or []))

print(f"время: {dt:.0f} с")
print("OK — API соответствует ТЗ" if not fail else "ОШИБКИ:\n- " + "\n- ".join(fail))
sys.exit(1 if fail else 0)
