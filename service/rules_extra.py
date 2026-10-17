"""Дополнительные правила на открытой модели OWLv2 (owl.py) — поверх правил engine.py.

Подключены в analyzer.analyze и tools/evaluate.py через find_extra(frames, dets).
"""
import os
import threading
import time

import numpy as np

from analyzer import Event
from engine import persons, runs, table_box, rule_door_open_auto

OWL_BUDGET_SEC = float(os.environ.get("OWL_BUDGET_SEC", 90))   # потолок времени на OWLv2 в одном видео
CASH_TH = 0.15   # обучающие: при приходе следующего клиента X-4 0.20–0.29, X-38 0.20–0.23; нормы ≤ 0.10
DRINK_TH = 0.22  # «a coffee cup» на столе, медиана 3 замеров через 2 с: X-46 0.40; нормы ≤ 0.11 (X-25), др. нарушения ≤ 0.09
DRINK_FULL_TH = 0.25
_tl = threading.local()   # потолок времени OWLv2 на одно видео (своё у каждой задачи — WORKERS > 1)


def _owl_ok():
    return time.monotonic() < getattr(_tl, "deadline", 0.0)


def desk_rect(box):
    return [max(0, box[0] - 0.05), max(0, box[1] - 0.04), min(1, box[2] + 0.05), min(1, box[3] + 0.02)]


def desk_crop(img, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = desk_rect(box)
    return img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


def locate(frames, ev, box, query):
    """Рамка найденного предмета на кадрах-доказательствах (в долях всего кадра) — для подсветки."""
    from owl import best_box
    r = desk_rect(box) if box else [0.0, 0.0, 1.0, 1.0]
    by_t = {round(f.t, 3): f for f in frames}
    for t in ev.evidence_t:
        f = by_t.get(round(t, 3))
        if f is None:
            continue
        img = desk_crop(f.hires(), box) if box else f.hires()
        _, (x1, y1, x2, y2) = best_box(img, query)
        sw, sh = r[2] - r[0], r[3] - r[1]
        ev.boxes.append((f.t, [r[0] + x1 * sw, r[1] + y1 * sh, r[0] + x2 * sw, r[1] + y2 * sh]))
    return ev


def rule_cash_for_next_client(frames, dets, n=3, th=CASH_TH):
    """Новый клиент подходит к кассе, а на столе уже лежит пачка денег (прошлого клиента не убрали в сейф).
    Визит = >= 2 человек подряд; смотрим первые n кадров каждого визита, кроме первого."""
    from owl import score
    box = table_box(dets)
    if box is None:
        return []
    visits = runs([len(persons(d)) >= 2 for d in dets], 2)
    ev = []
    for s, e in visits[1:]:
        if not _owl_ok():
            break
        idx = list(range(s, min(e + 1, s + n)))
        sc = [float(score(desk_crop(frames[i].hires(), box), ["a stack of banknotes"])[0]) for i in idx]
        if np.median(sc) >= th:
            ev.append(locate(frames, Event("cash_exposed", frames[s].t,
                            f"На {frames[s].t:.0f} с к кассе подходит следующий клиент, а на столе уже лежат "
                            "наличные (пачка купюр) — деньги предыдущей операции не убраны в место хранения.",
                            [frames[s].t, frames[idx[-1]].t]), box, "a stack of banknotes"))
    return ev


def rule_drink_on_desk(frames, dets, every=2.0, w=3, th=DRINK_TH):
    """Стакан/кружка с напитком стоит на рабочем столе (рядом с наличными) — устойчиво, не мелькнул."""
    from owl import score
    box = table_box(dets)
    if box is None:   # стол не найден (другой ракурс): смотрим весь кадр, порог выше — нормы по кадру ≤ 0.16
        th = max(th, DRINK_FULL_TH)
    idx, nxt = [], 0.0
    for i, f in enumerate(frames):
        if f.t >= nxt:
            idx.append(i)
            nxt = f.t + every
    sc = []
    for i in idx:
        if not _owl_ok():
            break
        img = desk_crop(frames[i].hires(), box) if box else frames[i].hires()
        sc.append(float(score(img, ["a coffee cup"])[0]))
        if len(sc) >= w and np.median(sc[-w:]) >= th:
            k0 = len(sc) - w   # момент появления — первый замер окна, где стакан уже виден
            j = idx[next(k for k in range(k0, len(sc)) if sc[k] >= th)]
            return [locate(frames, Event("food_drink", frames[j].t,
                          f"С {frames[j].t:.0f} с на рабочем столе кассира стоит стакан с напитком "
                          "рядом с местом работы с наличными.",
                          [frames[j].t, frames[i].t]), box, "a coffee cup")]
    return []


def rule_enter_open_door(frames, dets, gap=3.0):
    """Дверь кассы осталась открытой, и после этого в кассу входит ещё один человек.
    Организаторы отмечают именно момент входа (X-30: дверь открыта с 3 с, посторонний вошёл на 13 с)."""
    door = rule_door_open_auto(frames, dets)
    if not door:
        return []
    t0 = door[0].t
    cnt = [len(persons(d)) for d in dets]
    for i in range(1, len(frames)):
        if frames[i].t < t0 + gap:
            continue
        if cnt[i] > max(cnt[max(0, i - 3):i]) and cnt[i] >= 2 and (i + 1 >= len(cnt) or cnt[i + 1] >= 2):
            return [Event("outsider", frames[i].t,
                          f"Дверь кассы открыта с {t0:.0f} с, и на {frames[i].t:.0f} с через неё в кассу входит "
                          "ещё один человек, пока кассир работает с наличными.",
                          [frames[i].t, frames[min(i + 1, len(frames) - 1)].t])]
    return []


EXTRA_FN = [rule_cash_for_next_client, rule_drink_on_desk, rule_enter_open_door]


LEAVE_TYPES = {"computer_unlocked", "hasty_exit", "left_unattended"}
LEAVE_SHIFT_SEC = 2.0   # разметка ставит НАЧАЛО ухода (встала / отошла от сейфа), детектор видит уход на ~2 с позже:
                        # X-11 0, X-15 +3, X-24 +1, X-26 +4, X-28 +3 с → после сдвига 5 из 6 в пределах 2 с


def calibrate(events, frames):
    """Поправка таймкода событий «кассир ушёл» к моменту начала ухода (кадры-доказательства не трогаем)."""
    for e in events:
        if e.type in LEAVE_TYPES and frames:
            t = max(frames[0].t, e.t - LEAVE_SHIFT_SEC)
            e.t = min(frames, key=lambda f: abs(f.t - t)).t
            if e.t not in e.evidence_t:
                e.evidence_t = [e.t] + list(e.evidence_t)[:2]
    return events


def find_extra(frames, dets, deadline=None):
    _tl.deadline = min(time.monotonic() + OWL_BUDGET_SEC, deadline or float("inf"))
    n = min(len(frames), len(dets))
    out = []
    for fn in EXTRA_FN:
        out.extend(fn(frames[:n], dets[:n]))
    return out
