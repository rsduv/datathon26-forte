"""Сервис проверки кассовой дисциплины по видео (DATATHON'26, кейс ForteBank).

Запуск:   cd service && py -3.11 -m uvicorn app:app --host 0.0.0.0 --port 8000
Ключ API: переменная окружения API_KEY (по умолчанию — dev-key, сменить перед сдачей!).
"""
import json
import os
import time
import tempfile
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from analyzer import analyze

API_KEY = os.environ.get("API_KEY", "dev-key")
MAX_BYTES = 200 * 1024 * 1024
VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
TOTAL_BUDGET_SEC = float(os.environ.get("TOTAL_BUDGET_SEC", 240))   # от загрузки до ответа, лимит ТЗ — 300 с
WORKDIR = Path(tempfile.gettempdir()) / "cashcheck"
WORKDIR.mkdir(exist_ok=True)

def _gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


app = FastAPI(title="Cash discipline check")
jobs = {}                                  # job_id -> dict
lock = threading.Lock()
pool = ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS") or (2 if _gpu() else 1)))   # CPU: 1; GPU: 2+ видео одновременно


def _warmup():   # модели грузим при старте, а не на первом видео жюри
    try:
        from detector import _load
        from owl import _load as owl_load
        _load()
        owl_load()
    except Exception:
        traceback.print_exc()


threading.Thread(target=_warmup, daemon=True).start()


def check_key(key):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


JOBS_DIR = WORKDIR / "jobs"   # готовые результаты на диске: переживают перезапуск сервиса
JOBS_DIR.mkdir(exist_ok=True)


def run_job(job_id, path, submitted):
    try:   # лимит ТЗ считается от загрузки: ожидание в очереди входит в бюджет
        result = analyze(str(path), deadline=submitted + TOTAL_BUDGET_SEC)
        with lock:
            jobs[job_id].update(status="done", **result)
    except Exception as e:
        traceback.print_exc()
        with lock:
            jobs[job_id].update(status="failed", error=str(e))
    finally:
        path.unlink(missing_ok=True)
        with lock:
            snapshot = dict(jobs[job_id])
        (JOBS_DIR / f"{job_id}.json").write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")


async def submit(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")
    job_id = uuid.uuid4().hex[:8]
    suffix = Path(file.filename).suffix.lower()
    path = WORKDIR / f"{job_id}{suffix if suffix in VIDEO_EXT else '.mov'}"   # имя файла клиента не используем
    size = 0
    try:
        with path.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(status_code=413, detail="file too large")
                out.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    with lock:
        jobs[job_id] = {"job_id": job_id, "status": "processing"}
    pool.submit(run_job, job_id, path, time.monotonic())
    return {"job_id": job_id, "status": "processing"}


def get_job(job_id):
    with lock:
        job = jobs.get(job_id)
        if job is not None:
            return dict(job)
    saved = JOBS_DIR / f"{Path(job_id).name}.json"
    if saved.exists():
        return json.loads(saved.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="job not found")


# ---- публичный API для жюри (X-API-Key) -------------------------------------
@app.post("/api/v1/analyze")
async def api_analyze(file: UploadFile | None = File(None), x_api_key: str = Header(None)):
    check_key(x_api_key)   # сначала ключ (401), потом проверка файла
    if file is None:
        raise HTTPException(status_code=400, detail="file is required")
    return await submit(file)


@app.get("/api/v1/jobs/{job_id}")
def api_job(job_id: str, x_api_key: str = Header(None)):
    check_key(x_api_key)
    return get_job(job_id)


# ---- веб-интерфейс (без ключа, по ТЗ авторизация в UI не нужна) -------------
@app.post("/ui/analyze")
async def ui_analyze(file: UploadFile = File(...)):
    return await submit(file)


@app.get("/ui/jobs/{job_id}")
def ui_job(job_id: str):
    return get_job(job_id)


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"ok": True}
