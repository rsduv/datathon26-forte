@echo off
chcp 65001 >nul
rem Запуск сервиса одним двойным кликом (Windows).
rem Ключ для жюри задаётся здесь; поменять перед сдачей.
cd /d "%~dp0"
if "%API_KEY%"=="" set API_KEY=dev-key
echo Устанавливаю зависимости (первый раз долго)...
py -3.11 -m pip install -q -r requirements.txt
echo.
echo Сервис: http://localhost:8000   API-ключ: %API_KEY%
start "" http://localhost:8000
py -3.11 -m uvicorn app:app --host 0.0.0.0 --port 8000
pause
