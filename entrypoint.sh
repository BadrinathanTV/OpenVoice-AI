#!/bin/bash
set -e

# Start the Backend in the background
echo "Starting Backend..."
cd /app/backend
uv run uvicorn src.api.server:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start the Frontend
echo "Starting Frontend..."
cd /app/frontend
npm run preview -- --host 0.0.0.0 --port 5173 &
FRONTEND_PID=$!

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
