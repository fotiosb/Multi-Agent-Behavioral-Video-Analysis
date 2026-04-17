@echo off
echo Starting SentinelIQ...

cd /d %~dp0

echo Starting backend...
start "SentinelIQ Backend" cmd /k "venv\Scripts\activate && uvicorn backend.app.main:app --reload --port 8000"

timeout /t 3 /nobreak >nul

echo Starting frontend...
cd frontend
start "SentinelIQ Frontend" cmd /k "npm run dev"

echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo.
pause
