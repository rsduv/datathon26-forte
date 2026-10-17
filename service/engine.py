"""Правила: кадры + детекции -> события (Event).

Работает на списке frames (analyzer.Frame) и dets (list[list[dict]] из detector.detect, по кадру).
Отдельно от детектора, чтобы подбирать пороги на закэшированных детекциях (tools/evaluate.py).
"""
import cv2
import numpy as np

from analyzer import Event

# --- люди -------------------------------------------------------------------
PERSON_SCORE = 0.9         # люди за стеклом (соседнее помещение) детектятся с 0.6–0.93
PERSON_MIN_BOTTOM = 0.38   # отражения в стеклянной двери «висят» выше пола
PERSON_MIN_H = 0.12


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def persons(det):
    ps = sorted((d for d in det if d["label"] == "person" and d["score"] >= PERSON_SCORE
                 and d["box"][3] >= PERSON_MIN_BOTTOM and d["box"][3] - d["box"][1] >= PERSON_MIN_H),
                key=lambda d: -d["score"])
    keep = []
    for d in ps:   # один человек = одна рамка
        if all(iou(d["box"], k["box"]) < 0.4 for k in keep):
            keep.append(d)
    return keep


def runs(mask, min_len):
    """Отрезки подряд идущих True длиной >= min_len -> [(start, end_inclusive)]."""
    out, s = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = i
        elif not m and s is not None:
            if i - s >= min_len:
                out.append((s, i - 1))
            s = None
    return out


# --- правила ----------------------------------------------------------------
def rule_outsider(frames, dets, min_people=3, min_len=2, tall=0.22):
    """>=3 человек, из них >=2 стоят/в полный рост (кассир + лишний); сидящий клиент и люди за стеклом ниже."""
    ps = [persons(d) for d in dets]
    cnt = [len(p) for p in ps]
    ok = [len(p) >= min_people and sum(d["box"][3] - d["box"][1] >= tall for d in p) >= 2 for p in ps]
    ev = []
    for s, e in runs(ok, min_len):
        ev.append(Event("outsider", frames[s].t,
                        f"В кассовой зоне одновременно находятся {max(cnt[s:e + 1])} человека "
                        f"({frames[s].t:.0f}–{frames[e].t:.0f} с).",
                        [frames[s].t, frames[min(e, s + 1)].t]))
    return ev


def screen_bright(img, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    c = img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
    if c.size == 0:
        return 0.0
    v = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[..., 2]
    return float((v > 200).mean())


def rule_computer_unlocked(frames, dets, bright_th=0.12, min_len=1):
    """Кассир ушёл (в кадре никого), а экран ноутбука светится открытым рабочим окном."""
    present = [len(persons(d)) > 0 for d in dets]
    if not any(present):
        return []
    laptops = [d["box"] for det in dets for d in det if d["label"] == "laptop" and d["score"] >= 0.5]
    if not laptops:
        return []
    box = np.median(np.array(laptops), axis=0).tolist()
    first = present.index(True)
    ev = []
    last = len(present) - 1
    for s, e in runs([not p for p in present], min_len):
        if s <= first or (e - s + 1 < 3 and e != last):   # короткий пропуск детекции — не уход
            continue
        # экран меряем по рамке ноутбука, найденной в самих кадрах ухода (усреднённая рамка
        # на другом ракурсе может съехать на светлую стену); если ноутбука там не видно — по медиане
        br = []
        for i in range(s, e + 1):
            here = [d["box"] for d in dets[i] if d["label"] == "laptop" and d["score"] >= 0.5]
            if here:
                br.append(max(screen_bright(frames[i].img, b) for b in here))
        if not br and len(laptops) >= 5:
            br = [screen_bright(frames[i].img, box) for i in range(s, e + 1)]
        if br and np.median(br) >= bright_th:
            t_leave = leave_moment(frames, dets, s)
            ev.append(Event("computer_unlocked", t_leave,
                            "Кассир покинул кадр, при этом экран рабочего ноутбука остаётся включённым "
                            "с открытым рабочим окном (не заблокирован).",
                            [t_leave, frames[s].t]))
            break
    return ev


def leave_moment(frames, dets, s_absent):
    """Момент, когда кассир отходит от рабочего места: последний кадр, где человек ещё на «своём» месте
    (место = медиана рамки кассира до ухода; кассир = самый крупный человек)."""
    seat = []
    for i in range(s_absent):
        p = persons(dets[i])
        if p:
            seat.append(max(p, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))["box"])
    if not seat:
        return frames[s_absent].t
    home = np.median(np.array(seat), axis=0).tolist()
    for i in range(s_absent - 1, -1, -1):
        if any(iou(d["box"], home) >= 0.3 for d in persons(dets[i])):
            return frames[min(i + 1, s_absent)].t
    return frames[s_absent].t


DESK_ITEMS = {"laptop", "suitcase", "tv", "keyboard", "mouse", "cell phone"}   # ноутбук, счётная машинка, сейф


def table_box(dets):
    """Рабочий стол = объединение медианных рамок предметов на столе (ноутбук, счётная машинка, сейф).
    `dining table` не годится: детектор часто берёт белый стол на переднем плане."""
    by = {}
    for det in dets:
        for d in det:
            if d["label"] in DESK_ITEMS and d["score"] >= 0.5:
                by.setdefault(d["label"], []).append(d["box"])
    if not by:
        return None
    meds = np.array([np.median(np.array(v), axis=0) for v in by.values() if len(v) >= 3] or
                    [np.median(np.array(v), axis=0) for v in by.values()])
    x1, y1 = meds[:, 0].min(), meds[:, 1].min()
    x2, y2 = meds[:, 2].max(), meds[:, 3].max()
    return [x1, y1, x2, y2 + 0.5 * (y2 - y1)]


def green_on_table(img, box, pad=0.03):
    """Доля кадра, занятая самым крупным зелёным пятном на столе (инкассаторская сумка мятного цвета)."""
    h, w = img.shape[:2]
    x1, y1 = max(0, int((box[0] - pad) * w)), max(0, int((box[1] - pad) * h))
    x2, y2 = min(w, int((box[2] + pad) * w)), min(h, int((box[3] + pad) * h))
    c = img[y1:y2, x1:x2]
    if c.size == 0:
        return 0.0
    m = cv2.inRange(cv2.cvtColor(c, cv2.COLOR_BGR2HSV), (35, 35, 70), (95, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    return float(st[1:, cv2.CC_STAT_AREA].max()) / (h * w) if n > 1 else 0.0


def rule_bag_left(frames, dets, area_th=0.001, min_len=2):
    """Кассир ушёл (в кадре никого), а на столе осталась инкассаторская сумка (X-15; норма — X-12)."""
    present = [len(persons(d)) > 0 for d in dets]
    if not any(present):
        return []
    box = table_box(dets)
    if box is None:
        return []
    first = present.index(True)
    for s, e in runs([not p for p in present], min_len):
        if s <= first:
            continue
        areas = [green_on_table(frames[i].img, box) for i in range(s, e + 1)]
        if np.median(areas) >= area_th:
            t_leave = leave_moment(frames, dets, s)
            mid = frames[(s + e) // 2].t
            return [Event("left_unattended", t_leave,
                          "Кассир покинул рабочее место, а инкассаторская сумка осталась на столе без присмотра "
                          f"({frames[s].t:.0f}–{frames[e].t:.0f} с), не убрана в сейф.",
                          [t_leave, mid])]
    return []


DOOR_ZONE = (0.03, 0.02, 0.30, 0.25)   # верх стеклянной двери (слева сверху на всех 3 ракурсах); кассир её не загораживает


def _gray(img, zone=None):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if zone is None:
        return g
    h, w = g.shape
    x1, y1, x2, y2 = zone
    return g[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


def rule_door_open(frames, dets, diff_th=32.0, min_sec=5.0, light_th=0.2):
    """Дверь остаётся открытой: зона двери отличается от первого кадра (дверь закрыта) дольше min_sec.
    Короткие всплески — человек проходит — не считаются. Выключенный свет (меняется весь кадр) отсекаем."""
    if len(frames) < 3:
        return []
    ref, ref_all = _gray(frames[0].img, DOOR_ZONE), _gray(frames[0].img).mean()
    open_ = []
    for f in frames:
        g_all = _gray(f.img).mean()
        light_changed = abs(g_all - ref_all) / (ref_all + 1e-6) > light_th
        open_.append(not light_changed and float(np.abs(_gray(f.img, DOOR_ZONE) - ref).mean()) >= diff_th)
    dt = (frames[-1].t - frames[0].t) / max(1, len(frames) - 1)
    for s, e in runs(open_, max(2, int(round(min_sec / max(dt, 1e-6))))):
        return [Event("door_open", frames[s].t,
                      f"Дверь кассового помещения открыта с {frames[s].t:.0f} с и остаётся открытой "
                      f"{frames[e].t - frames[s].t + dt:.0f} с; за ней не закрыли.",
                      [frames[s].t, frames[min(e, s + 3)].t])]
    return []


def rule_door_open_auto(frames, dets, rows=6, cols=6, top=0.55, diff_th=30.0, min_sec=5.0,
                        min_cells=3, light_th=0.2):
    """Без фиксированной зоны (для другого ракурса): верх кадра делим на клетки; клетка «изменена»,
    если отличается от первого кадра и её не загораживает человек. Дверь открыта, если >= min_cells
    клеток одновременно изменены дольше min_sec."""
    if len(frames) < 3:
        return []
    h, w = frames[0].img.shape[:2]
    ys = np.linspace(0, int(h * top), rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    ref, ref_all = _gray(frames[0].img), _gray(frames[0].img).mean()
    changed = np.zeros((len(frames), rows, cols), bool)
    for i, f in enumerate(frames):
        g = _gray(f.img)
        if abs(g.mean() - ref_all) / (ref_all + 1e-6) > light_th:
            continue
        busy = np.zeros((h, w), bool)
        for d in dets[i]:
            if d["label"] == "person" and d["score"] >= 0.4:
                x1, y1, x2, y2 = d["box"]
                busy[max(0, int(y1 * h) - 5):int(y2 * h) + 5, max(0, int(x1 * w) - 5):int(x2 * w) + 5] = True
        for r in range(rows):
            for c in range(cols):
                sl = (slice(ys[r], ys[r + 1]), slice(xs[c], xs[c + 1]))
                if busy[sl].mean() < 0.2:
                    changed[i, r, c] = float(np.abs(g[sl] - ref[sl]).mean()) >= diff_th
    dt = (frames[-1].t - frames[0].t) / max(1, len(frames) - 1)
    need = max(2, int(round(min_sec / max(dt, 1e-6))))
    # клетка «устойчиво изменена» в кадре i, если изменена во всех кадрах окна [i, i+need)
    n = len(frames)
    for i in range(n - need + 1):
        stable = changed[i:i + need].all(axis=0)
        if stable.sum() >= min_cells:
            return [Event("door_open", frames[i].t,
                          f"С {frames[i].t:.0f} с часть помещения у входа остаётся изменённой более "
                          f"{min_sec:.0f} с при отсутствии людей в этой зоне — дверь кассового помещения, "
                          f"вероятно, не закрыта.",
                          [frames[i].t, frames[min(n - 1, i + need - 1)].t])]
    return []


def _biggest(ps):
    return max(ps, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))["box"]


def rule_quick_exit(frames, dets, max_walk_sec=3.0, min_present=5):
    """Кассир встаёт и сразу выходит из кабинета (≤ max_walk_sec), не задерживаясь у сейфа.
    На обучающих видео в норме от вставания до выхода 4–14 с (убирает ценности в сейф, запирает его),
    при нарушении 2–3 с (X-13, X-15, X-24, X-26). Учитываем только уход до конца видео или надолго."""
    present = [len(persons(d)) > 0 for d in dets]
    n = len(present)
    if sum(present) < min_present:
        return []
    first = present.index(True)
    exits = [(s, e) for s, e in runs([not p for p in present], 1)
             if s > first and sum(present[:s]) >= min_present and (e == n - 1 or e - s >= 4)]
    if not exits:
        return []
    s = exits[-1][0]
    # рабочее место = где кассир был в начале видео (первые кадры с человеком)
    early = [_biggest(persons(dets[i])) for i in range(s) if persons(dets[i])][:8]
    seat = np.median(np.array(early), axis=0).tolist()
    on_seat = [i for i in range(s) if any(iou(d["box"], seat) >= 0.3 for d in persons(dets[i]))]
    if not on_seat:
        return []
    i_seat = on_seat[-1]
    dt = frames[1].t - frames[0].t if n > 1 else 1.0
    walk = frames[s].t - frames[i_seat].t - dt
    if walk > max_walk_sec + 0.25 * dt:      # допуск на дробный fps (29.9 → 3.003 с)
        return []
    t_seat = frames[min(i_seat + 1, s)].t
    return [Event("hasty_exit", t_seat,
                  f"Кассир встаёт с рабочего места на {t_seat:.0f} с и через {max(walk, 1):.0f} с покидает кабинет, "
                  "не задерживаясь у сейфа. Перед уходом ценности и документы убираются в сейф, сейф запирается — "
                  "проверьте по кадрам, что на столе ничего не осталось и сейф заперт.",
                  [t_seat, frames[s].t])]


BAG_LABELS = ("handbag", "backpack")


def rule_personal_items(frames, dets, score=0.5, static_share=0.3, min_len=2):
    """Личная сумка появляется на новом месте и лежит там (X-40). Детектор подписывает «сумкой» и приборы
    на столе (телефон, машинка) — они стоят на одном месте почти всё видео, их отбрасываем как «статичные»."""
    n = len(dets)
    boxes = [[d["box"] for d in det if d["label"] in BAG_LABELS and d["score"] >= score] for det in dets]
    flat = [b for bs in boxes for b in bs]
    static = []
    for b in flat:   # место, где «сумка» видна в >= static_share кадров, — это прибор, а не сумка
        if sum(any(iou(b, x) >= 0.5 for x in bs) for bs in boxes) >= static_share * n and \
                all(iou(b, s) < 0.5 for s in static):
            static.append(b)
    novel = [[b for b in bs if all(iou(b, s) < 0.3 for s in static)] for bs in boxes]
    for i in range(n - min_len + 1):
        for b in novel[i]:
            if all(any(iou(b, x) >= 0.4 for x in novel[j]) for j in range(i + 1, i + min_len)):
                return [Event("personal_items", frames[i].t,
                              f"На {frames[i].t:.0f} с в кассовой зоне появляется сумка, не относящаяся к оборудованию, "
                              f"и остаётся на месте — вероятно, личные вещи кассира.",
                              [frames[i].t, frames[min(n - 1, i + min_len)].t])]
    return []


def rule_short_visit(frames, dets, max_sec=10.0, after=2, gap=1):
    """Посетитель (клиент/инкассатор) уходит слишком быстро: визит <= max_sec, после него кассир один.
    В норме на обучающих видео визит длится > 15 с (проверка документа, пересчёт при клиенте, сейф);
    X-17 — инкассатор отдал сумку и ушёл через ~8 с."""
    n = len(dets)
    two = [len(persons(d)) >= 2 for d in dets]
    for i in range(1, n - 1):      # склеиваем одиночные пропуски детекции
        if not two[i] and two[i - 1] and any(two[i + 1:i + 1 + gap]):
            two[i] = True
    dt = (frames[-1].t - frames[0].t) / max(1, n - 1)
    for s, e in runs(two, 4):          # >= 4 кадров: одиночные «вторые люди» — шум детектора
        dur = frames[e].t - frames[s].t + dt
        alone_after = e + after < n and all(len(persons(dets[j])) == 1 for j in range(e + 1, e + 1 + after))
        if dur <= max_sec and alone_after and s > 2:   # визит в первые секунды — часто дрожание детекции
            return [Event("short_visit", frames[s].t,
                          f"Посетитель (клиент или инкассатор) подходит к кассе на {frames[s].t:.0f} с и уходит через "
                          f"{dur:.0f} с — за это время проверка документа и пересчёт ценностей в его присутствии "
                          f"вряд ли проведены. Проверьте фрагмент.",
                          [frames[s].t, frames[e].t])]
    return []


# rule_personal_items выключено: на обучающих видео 3 ложных из 19 (сумки клиентов/пачки денег)
RULES_FN = [rule_outsider, rule_computer_unlocked, rule_bag_left, rule_door_open_auto, rule_quick_exit,
            rule_short_visit]


def dedupe(events, window=10.0):
    """Один уход кассира = одно событие «ушёл»: если уже есть конкретное (ноутбук/сумка), общее не добавляем."""
    leave_specific = [e for e in events if e.type in ("computer_unlocked",) or
                      (e.type == "left_unattended" and "сумка" in e.observed)]
    out = []
    for e in events:
        if e.type == "hasty_exit" and any(abs(e.t - x.t) <= window for x in leave_specific):
            continue
        out.append(e)
    return out


def normalize_brightness(frames, target=240.0):
    """Камеры отделений по-разному экспонируют кадр: приводим яркость видео к эталону обучающих роликов
    (99-й перцентиль яркости ≈ 240), чтобы пороги яркости/цвета работали на любой камере."""
    import copy
    vals = [np.percentile(cv2.cvtColor(f.img, cv2.COLOR_BGR2HSV)[..., 2], 99) for f in frames[::max(1, len(frames) // 10)]]
    p99 = float(np.median(vals)) if vals else target
    if p99 <= 0 or abs(p99 - target) < 8:
        return frames
    gain = min(2.0, target / p99)
    out = []
    for f in frames:
        g = copy.copy(f)
        g.img = cv2.convertScaleAbs(f.img, alpha=gain, beta=0)
        out.append(g)
    return out


def find_events(frames, dets):
    n = min(len(frames), len(dets))
    frames, dets = normalize_brightness(frames[:n]), dets[:n]
    ev = []
    for fn in RULES_FN:
        ev.extend(fn(frames, dets))
    return dedupe(ev)
