#!/usr/bin/env bash
# Запуск сервиса на арендованном GPU-сервере (RunPod, шаблон «RunPod PyTorch», открыть HTTP-порт 7860).
# В веб-терминале пода:
#   (ключ жюри создаётся сам и печатается в конце; задать свой — export API_KEY=...)
#   curl -fsSL <raw-ссылка на этот файл> | bash      (или скопировать файл и: bash runpod_start.sh)
# Ссылка для жюри: https://<POD_ID>-7860.proxy.runpod.net
set -euo pipefail
: "${API_KEY:=$(cat /workspace/api_key.txt 2>/dev/null || python -c 'import secrets;print("fb-"+secrets.token_urlsafe(24))')}"
echo "$API_KEY" > /workspace/api_key.txt   # ключ жюри: создаётся один раз и переживает перезапуск
GIT_URL=${GIT_URL:-https://github.com/rsduv/datathon26-forte.git}
BRANCH=${BRANCH:-main}
cd /workspace
[ -d datathon26 ] || git clone --depth 1 -b "$BRANCH" "$GIT_URL" datathon26
cd datathon26 && git pull --ff-only || true
export API_KEY
cd service
apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 ffmpeg >/dev/null
# torch уже стоит в шаблоне с CUDA — ставим остальное, не трогая его
grep -v -E '^torch(vision)?' requirements.txt > /tmp/req.txt
pip install -q -r /tmp/req.txt
python -c "import torch; print('GPU:', torch.cuda.is_available())"
python -c "from detector import _load; _load(); from owl import _load as o; o()"   # веса один раз
export MAX_FRAMES=${MAX_FRAMES:-600} WORKERS=${WORKERS:-2}
pkill -f "uvicorn app:app" || true   # повторный запуск = обновление версии
sleep 1
nohup uvicorn app:app --host 0.0.0.0 --port 7860 > /workspace/service.log 2>&1 &
sleep 5 && curl -s localhost:7860/health && echo && echo "готово: https://${RUNPOD_POD_ID:-<POD_ID>}-7860.proxy.runpod.net"
echo "API-ключ для жюри: $API_KEY"
