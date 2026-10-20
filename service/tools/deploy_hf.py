"""Выкладка сервиса в публичный Hugging Face Space (Docker).

Сначала войти один раз: py -3.11 -c "from huggingface_hub import login; login()" (токен с правом Write).
Затем: py -3.11 service/tools/deploy_hf.py   (необязательно: HF_SPACE=аккаунт/имя, API_KEY=свой ключ)
Папка по умолчанию — service/ (берутся только файлы сервиса). Повторный запуск обновляет Space, ссылка та же.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO = os.environ.get("HF_SPACE", "romikosh/cash-discipline-check")
SERVICE = Path(__file__).resolve().parent.parent
FILES = ["Dockerfile", ".dockerignore", "requirements.txt", "app.py", "analyzer.py", "detector.py",
         "engine.py", "rules.py", "rules_extra.py", "owl.py", "static"]
HEADER = """---
title: Cash Discipline Check
emoji: 🏦
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Проверка кассовой дисциплины по видео (DATATHON'26, кейс ForteBank)

Загрузите .mov — сервис найдёт признаки возможных нарушений, покажет таймкод, кадры-доказательства
и норматив РК. API: `POST /api/v1/analyze` (поле `file`, заголовок `X-API-Key`) → `GET /api/v1/jobs/{job_id}`.
Все модели работают внутри контейнера, без облачных ИИ-сервисов. Бесплатный CPU — видео обрабатывается 1–2 мин.
Код и описание: https://github.com/rsduv/datathon26-forte
"""

api = HfApi(token=os.environ.get("HF_TOKEN"))   # без HF_TOKEN — сохранённый вход (login())
print("аккаунт:", api.whoami()["name"])
with tempfile.TemporaryDirectory() as tmp:
    for f in FILES:
        src = SERVICE / f
        (shutil.copytree if src.is_dir() else shutil.copy)(src, Path(tmp) / f)
    (Path(tmp) / "README.md").write_text(HEADER, encoding="utf-8")
    api.create_repo(REPO, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    if os.environ.get("API_KEY"):
        api.add_space_secret(REPO, "API_KEY", os.environ["API_KEY"])
    api.upload_folder(repo_id=REPO, repo_type="space", folder_path=tmp,
                      commit_message=sys.argv[1] if len(sys.argv) > 1 else "deploy")
print(f"готово: https://huggingface.co/spaces/{REPO}")
print(f"сервис: https://{REPO.replace('/', '-')}.hf.space")
