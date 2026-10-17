# Сервис проверки кассовой дисциплины

## Запуск
**Windows, одним кликом:** `service\run.bat` → откроется http://localhost:8000 (ключ API по умолчанию `dev-key`,
свой: `set API_KEY=...` перед запуском).

Вручную:
```
cd service
py -3.11 -m pip install -r requirements.txt
$env:API_KEY="<секрет>"          # cmd: set API_KEY=<секрет>
py -3.11 -m uvicorn app:app --host 0.0.0.0 --port 8000
```
Docker / Hugging Face Spaces: `Dockerfile` (порт 7860, веса детектора скачиваются при сборке, GPU подхватывается сам).

**Проверка API как у жюри:** `py -3.11 tools/smoke_test.py <видео.mov> --url http://localhost:8000 --key dev-key`
→ ключ (401 без него), загрузка, опрос статуса, схема JSON, JPEG-кадры, время ≤ 5 мин.

## Архитектура
```
UI (static/index.html) ─┐                         ┌─ detector.py     Faster R-CNN MobileNetV3 (torchvision, BSD, COCO)
API /api/v1/* (X-API-Key)┴→ app.py ─ очередь ─→ analyzer.py ─┼─ engine.py       правила по детекциям → события (тип, таймкод, кадры)
                          (FastAPI, WORKERS)   кадры 1 fps   ├─ rules_extra.py  правила на OWLv2 (деньги / стакан на столе)
                                                             │    └─ owl.py     OWLv2 google/owlv2-base-patch16-ensemble (Apache-2.0)
                                                             └─ rules.py        тип → заголовок, «почему», норматив РК
```
- `app.py` — FastAPI: API для жюри (ключ) и UI (без ключа, по ТЗ). Очередь задач (`WORKERS`, по умолчанию 1 на CPU).
  Готовые результаты пишутся на диск → `GET /jobs/{id}` работает и после перезапуска. Модели прогреваются при старте.
- `analyzer.py` — чтение кадров подряд (seek на части .mov даёт чёрные кадры, X-11), 1 кадр/с,
  не больше `MAX_FRAMES` (600 на GPU, 60 на CPU — тогда кадры берутся реже) и не дольше `TIME_BUDGET_SEC` (200 с)
  на детекцию — страховка лимита 5 мин. Сборка ответа по схеме ТЗ (раздел 10).
- `engine.py` — правила поверх детекций (`RULES_FN`): `outsider`, `computer_unlocked`, `rule_bag_left` → `left_unattended`,
  `door_open` (`rule_door_open_auto`), `rule_quick_exit` → `hasty_exit`, `short_visit`. Перед правилами — выравнивание
  яркости; после — `dedupe` (общий `hasty_exit` не дублирует `computer_unlocked`/сумку). `rule_personal_items` есть,
  но выключено (3 ложных из 19). Новое правило → функция + `RULES_FN`.
- `rules_extra.py` — правила на OWLv2 (`find_extra`, общий потолок `OWL_BUDGET_SEC` = 90 с на видео):
  `cash_exposed` — «a stack of banknotes» на 3 кадрах вырезки стола при приходе каждого следующего клиента;
  `food_drink` — «a coffee cup» на вырезке стола каждые 2 с (медиана 3 замеров), если стол не найден — весь кадр с порогом выше.
- `owl.py` — загрузка OWLv2 и оценка кадра по текстовым запросам.
- `rules.py` — справочник типов и нормативных оснований (сверено дословно, см. `laws/LAWS.md`).
- `tools/` — дампы детекций и `evaluate.py` для подбора порогов на 42 роликах (40 уникальных), `smoke_test.py`.

## Скорость
GPU (ноутбучная RTX 3050): 1–13 с на видео, все кадры 1/с. CPU: примерно 20–90+ с на видео (детектор ~1.75 с/кадр,
не больше 60 кадров, плюс OWLv2 до 90 с). Лимит ТЗ — 5 мин.
