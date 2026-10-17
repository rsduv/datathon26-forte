"""Анализ видео: кадры → детекторы → нарушения.

Контракт: analyze(path) -> {"verdict": ..., "violations": [...]} в формате API из ТЗ (раздел 10).
Детекторы подключаются в DETECTORS; каждый получает список кадров и возвращает события.
"""
import base64
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from rules import RULES

SAMPLE_FPS = 1.0      # кадров в секунду (CPU: ~0.9 с на кадр, лимит 5 мин на видео)
MAX_SIDE = 640        # уменьшаем кадр для скорости
FULL_MAX_SIDE = int(os.environ.get("FULL_MAX_SIDE", 1920))   # до какого размера храним исходник для мелких предметов
# CPU: детектор ~1.75 с/кадр → 60 кадров ≈ 2 мин; на GPU можно все кадры при 1 fps
MAX_FRAMES = int(os.environ.get("MAX_FRAMES") or (600 if torch.cuda.is_available() else 60))
TIME_BUDGET_SEC = float(os.environ.get("TIME_BUDGET_SEC", 200))  # жёсткий потолок на детекцию
STATIC_DIFF = float(os.environ.get("STATIC_DIFF", 0))  # >0: пропуск детектора на неподвижных кадрах (выключено: tools/eval_static.py дал ложное)


@dataclass
class Frame:
    t: float            # секунда от начала видео
    img: np.ndarray     # BGR, уменьшенный (для детектора)
    full: bytes = None  # исходный кадр (JPEG), если видео крупнее MAX_SIDE — для мелких предметов (OWLv2)

    def hires(self, max_side=FULL_MAX_SIDE):
        """Кадр в лучшем доступном качестве (до max_side): видео высокого разрешения — мелкие предметы крупнее."""
        if self.full is None:
            return self.img
        return cv2.imdecode(np.frombuffer(self.full, np.uint8), cv2.IMREAD_COLOR)


@dataclass
class Event:
    type: str           # ключ из rules.RULES
    t: float            # таймкод события
    observed: str       # что именно видно на видео
    evidence_t: list = field(default_factory=list)  # секунды кадров-доказательств (1–3)
    boxes: list = field(default_factory=list)       # [(t, [x1,y1,x2,y2] в долях кадра)] — что обвести на кадрах


def read_frames(path, sample_fps=SAMPLE_FPS):
    """Читаем подряд (без seek — на части .mov seek отдаёт чёрные кадры)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видео")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    if duration > 0:   # лимит жюри 5 мин на видео: длинные ролики проверяем реже
        sample_fps = min(sample_fps, MAX_FRAMES / duration)
    step = max(1, int(round(fps / sample_fps)))
    frames, i = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, img = cap.retrieve()
            if ok and img is not None:
                h, w = img.shape[:2]
                s = MAX_SIDE / max(h, w)
                full = None
                if s < 1:
                    big = img
                    sf = FULL_MAX_SIDE / max(h, w)
                    if sf < 1:
                        big = cv2.resize(img, (int(w * sf), int(h * sf)), interpolation=cv2.INTER_AREA)
                    full = cv2.imencode(".jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
                    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                frames.append(Frame(t=i / fps, img=img, full=full))
        i += 1
    cap.release()
    if not frames:
        raise ValueError("В видео нет читаемых кадров")
    return frames


# что обвести на кадре-доказательстве, если правило само не дало рамки (классы детектора)
HIGHLIGHT = {"computer_unlocked": {"laptop"}, "outsider": {"person"}, "short_visit": {"person"}}
BOX_COLOR = (40, 40, 230)   # BGR: красный — предмет нарушения
PERSON_COLOR = (0, 165, 255)   # BGR: оранжевый — кассир в момент ухода
LEAVE_TYPES = {"computer_unlocked", "hasty_exit", "left_unattended"}


def draw_boxes(img, boxes, people=()):
    out = img.copy()
    h, w = out.shape[:2]
    th = max(2, round(min(h, w) / 160))
    for color, bs in ((PERSON_COLOR, people), (BOX_COLOR, boxes)):
        for x1, y1, x2, y2 in bs:
            cv2.rectangle(out, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), color, th)
    return out


def green_bag_box(img, table):
    """Рамка самого крупного зелёного пятна на столе (инкассаторская сумка, как в engine.green_on_table)."""
    h, w = img.shape[:2]
    x1, y1 = max(0, int((table[0] - 0.03) * w)), max(0, int((table[1] - 0.03) * h))
    x2, y2 = min(w, int((table[2] + 0.03) * w)), min(h, int((table[3] + 0.03) * h))
    m = cv2.inRange(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV), (35, 35, 70), (95, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return []
    k = 1 + int(st[1:, cv2.CC_STAT_AREA].argmax())
    bx, by, bw, bh = st[k, :4]
    if bw * bh < 0.0005 * h * w:
        return []
    return [[(x1 + bx) / w, (y1 + by) / h, (x1 + bx + bw) / w, (y1 + by + bh) / h]]


def median_box(dets, label, th=0.5):
    b = [d["box"] for det in dets for d in det if d["label"] == label and d["score"] >= th]
    return np.median(np.array(b), axis=0).tolist() if b else None


def evidence_boxes(e, t, det, img=None, table=None, laptop=None):
    """Рамки для кадра в момент t: свои рамки правила (OWLv2) или объекты детектора по типу нарушения."""
    own = [b for bt, b in e.boxes if abs(bt - t) < 0.6]
    if own or e.boxes:
        return own
    labels = HIGHLIGHT.get(e.type, set())
    out = [d["box"] for d in det if d["label"] in labels and d["score"] >= (0.9 if d["label"] == "person" else 0.5)]
    if not out and e.type == "left_unattended" and img is not None and table is not None:
        out = green_bag_box(img, table)   # сумка — по цвету (детектор путает счётную машинку с suitcase)
    if not out and e.type == "computer_unlocked" and laptop:
        out = [laptop]                     # в темноте ноутбук не детектится — берём его место за всё видео
    return out


def jpeg_b64(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---- детекторы -------------------------------------------------------------
# Правила — в engine.py (find_events); детектор — detector.py.

def analyze(path, deadline=None):
    """deadline — time.monotonic(), к которому нужен ответ (лимит ТЗ); детекция оставляет запас на правила."""
    deadline = deadline or (time.monotonic() + TIME_BUDGET_SEC + 40)
    from detector import detect   # ленивый импорт: engine импортирует Event отсюда
    from engine import find_events
    from rules_extra import calibrate, find_extra
    frames = read_frames(path)
    by_t = {round(f.t, 3): f for f in frames}

    def nearest(t):
        return min(frames, key=lambda f: abs(f.t - t))

    dets, t0, prev = [], time.monotonic(), None
    for f in frames:   # страховка лимита жюри (5 мин): не успели — анализируем то, что уже обработано
        if time.monotonic() - t0 > TIME_BUDGET_SEC or time.monotonic() > deadline - 60:
            break   # 60 с запаса — на OWLv2 и правила
        small = cv2.cvtColor(cv2.resize(f.img, (96, 160)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None and dets and float(np.abs(small - prev).mean()) < STATIC_DIFF:
            dets.append(dets[-1])   # камера неподвижна, в кадре ничего не изменилось — детекции те же
        else:
            dets.append(detect(f.img))
            prev = small
    events = calibrate(find_events(frames, dets) + find_extra(frames, dets, deadline=deadline - 10), frames)
    events.sort(key=lambda e: e.t)

    from engine import table_box
    table, laptop = table_box(dets), median_box(dets, "laptop")
    violations = []
    for e in events:
        rule = RULES[e.type]
        ts = list(dict.fromkeys(round(t, 1) for t in (e.evidence_t or [e.t])))   # без повторов одного кадра
        evidence = []
        for t in ts[:3]:
            f = by_t.get(round(t, 3)) or nearest(t)
            i = frames.index(f)
            det_i = dets[i] if i < len(dets) else []
            boxes = evidence_boxes(e, f.t, det_i, f.img, table, laptop)
            people = [d["box"] for d in det_i if d["label"] == "person" and d["score"] >= 0.9] if e.type in LEAVE_TYPES else []
            img = draw_boxes(f.img, boxes, people) if boxes or people else f.img
            evidence.append({"timestamp_sec": round(f.t, 1), "image_base64": jpeg_b64(img)})
        violations.append({
            "type": rule["title"],
            "timecode_sec": round(e.t, 1),
            "description": f"{e.observed} Почему: {rule['why']} Основание: {rule['source']}.",
            "evidence_frames": evidence,
        })

    return {
        "verdict": "violation" if violations else "no_violation",
        "violations": violations,
    }
