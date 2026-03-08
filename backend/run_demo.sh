#!/bin/bash
# Workaround for Miniconda libstdc++ version mismatch with system PortAudio
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTHONUNBUFFERED=1

echo "Starting OpenVoice AI Multi-Agent System..."
uv run python -u -m src.main
