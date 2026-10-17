# Образ для Hugging Face Spaces (Docker SDK) или любого сервера.
# На GPU-железе torch сам возьмёт CUDA; на CPU тоже работает (медленнее).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces запускает контейнер от пользователя с uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
# Веса детектора скачиваем при сборке: при обработке видео интернет не нужен (требование переносимости)
RUN python -c "from detector import _load; _load(); from owl import _load as o; o()"
# Дальше — строго офлайн: библиотеки не ходят в сеть даже за проверкой версий (работа во внутреннем контуре банка)
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# На GPU можно проверять больше кадров длинного видео; на CPU оставить 60
ENV WORKERS=1
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
