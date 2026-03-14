@echo off
echo ===================================================
echo 🚀 Starting OpenVoice AI Application...
echo ===================================================

:: Environment variables
set PYTHONUNBUFFERED=1
:: Note: LD_PRELOAD is Linux-specific and removed here.

:: 1. Backend Check and Start
if not exist "backend\" (
    echo ❌ Error: 'backend' directory not found.
    pause
    exit /b 1
)

if not exist "backend\.env" (
    echo ⚠️  Warning: backend\.env not found. Backend might not start correctly.
)

echo 📦 Starting Backend Server in a new window...
cd backend
set "ACCESS_LOG_FLAG=--no-access-log"
if "%OPENVOICE_ACCESS_LOG%"=="1" set "ACCESS_LOG_FLAG="
set "UVICORN_LOG_LEVEL=warning"
if not "%OPENVOICE_UVICORN_LOG_LEVEL%"=="" set "UVICORN_LOG_LEVEL=%OPENVOICE_UVICORN_LOG_LEVEL%"
start "OpenVoice Backend" cmd /k "uv run uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --log-level %UVICORN_LOG_LEVEL% %ACCESS_LOG_FLAG%"
cd ..

:: 2. Frontend Check and Start
if not exist "frontend\" (
    echo ❌ Error: 'frontend' directory not found.
    pause
    exit /b 1
)

echo 🎨 Starting Frontend Dev Server in a new window...
cd frontend
start "OpenVoice Frontend" cmd /k "npm run dev"
cd ..

echo.
echo ✨ Both services are starting in separate windows.
echo    - Backend: http://localhost:8000
echo    - Frontend: http://localhost:5173
echo.
echo Close the newly opened terminal windows to stop the services.
echo.
pause
