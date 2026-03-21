#!/bin/bash

# Configuration and Environment
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTHONUNBUFFERED=1

# ANSI color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting OpenVoice AI Application...${NC}"

# 1. Backend Check and Start
if [ ! -d "backend" ]; then
    echo -e "${RED}❌ Error: 'backend' directory not found.${NC}"
    exit 1
fi

if [ ! -f "backend/.env" ]; then
    echo -e "${RED}⚠️  Warning: backend/.env not found. Backend might not start correctly.${NC}"
fi

echo -e "${GREEN}📦 Starting Backend Server...${NC}"
cd backend
ACCESS_LOG_FLAG="--no-access-log"
if [ "${OPENVOICE_ACCESS_LOG:-0}" = "1" ]; then
    ACCESS_LOG_FLAG=""
fi
UVICORN_LOG_LEVEL="${OPENVOICE_UVICORN_LOG_LEVEL:-warning}"

ROOT_DIR="$(pwd)/.."
VENV_BIN="${ROOT_DIR}/.venv/bin"

if [ -x "${VENV_BIN}/uvicorn" ]; then
    PYTHONPATH="${PWD}:${PYTHONPATH}" "${VENV_BIN}/uvicorn" src.api.server:app --host 0.0.0.0 --port 8000 --log-level "${UVICORN_LOG_LEVEL}" ${ACCESS_LOG_FLAG} &
elif [ -x "${VENV_BIN}/python" ]; then
    PYTHONPATH="${PWD}:${PYTHONPATH}" "${VENV_BIN}/python" -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --log-level "${UVICORN_LOG_LEVEL}" ${ACCESS_LOG_FLAG} &
elif command -v uv >/dev/null 2>&1; then
    uv run uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --log-level "${UVICORN_LOG_LEVEL}" ${ACCESS_LOG_FLAG} &
else
    echo -e "${RED}❌ Error: no usable Python backend launcher found. Create .venv or install uv.${NC}"
    exit 1
fi
BACKEND_PID=$!
cd ..

# 2. Frontend Check and Start
if [ ! -d "frontend" ]; then
    echo -e "${RED}❌ Error: 'frontend' directory not found.${NC}"
    kill $BACKEND_PID
    exit 1
fi

echo -e "${GREEN}🎨 Starting Frontend Dev Server...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Function to handle cleanup on exit
cleanup() {
    echo -e "\n${BLUE}🛑 Shutting down OpenVoice AI...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

echo -e "${BLUE}✨ Both services are starting in the background.${NC}"
echo -e "   - Backend: http://localhost:8000"
echo -e "   - Frontend: http://localhost:5173"
echo -e "${BLUE}Press Ctrl+C to stop both services.${NC}"

# Keep the script running
wait
