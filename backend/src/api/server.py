"""
FastAPI server with WebSocket endpoints for the OpenVoice AI voice assistant.
Provides both voice (binary audio) and text chat modes.
"""

import asyncio
import base64
import json
from contextlib import suppress

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ─── Lazy-loaded heavy AI components ────────────────────────────────────────
# We delay imports of torch / ML models until they are actually needed so the
# server process starts fast and the /api/health endpoint works instantly.

_pipeline_instance = None
_pipeline_lock = asyncio.Lock()


def _build_pipeline():
    """Instantiate the heavy ML models (runs once, in-process)."""
    from src.vad.silero import SileroVAD
    from src.asr.whisper import ASRModel
    from src.tts.piper import TTSModel
    from src.api.web_pipeline import WebVoicePipeline

    print("[Server] Loading AI models...")
    vad = SileroVAD()
    asr = ASRModel()
    tts_models = {
        "CustomerCare": TTSModel(model_name="en_GB-alba-medium"),
        "Shopper": TTSModel(model_name="en_US-bryce-medium", speed=1.20),
        "OrderOps": TTSModel(model_name="en_US-hfc_female-medium"),
    }
    pipeline = WebVoicePipeline(vad=vad, asr=asr, tts_models=tts_models)
    print("[Server] AI models ready.")
    return pipeline


async def get_pipeline():
    """Thread-safe singleton accessor for the pipeline."""
    global _pipeline_instance
    if _pipeline_instance is None:
        async with _pipeline_lock:
            if _pipeline_instance is None:
                _pipeline_instance = await asyncio.get_event_loop().run_in_executor(
                    None, _build_pipeline
                )
    return _pipeline_instance


# ─── FastAPI App ────────────────────────────────────────────────────────────

app = FastAPI(
    title="OpenVoice AI",
    description="Real-time voice AI assistant API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REST Endpoints ────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "models_loaded": _pipeline_instance is not None}


@app.get("/api/agents")
async def list_agents():
    return {
        "agents": [
            {"name": "CustomerCare", "description": "General help, returns, refunds, policies", "color": "#6C63FF"},
            {"name": "Shopper", "description": "Product search and recommendations", "color": "#00C9A7"},
            {"name": "OrderOps", "description": "Order tracking and operations", "color": "#FF6B6B"},
        ]
    }


# ─── WebSocket: Voice Mode ─────────────────────────────────────────────────

@app.websocket("/ws/voice")
async def websocket_voice(ws: WebSocket):
    await ws.accept()
    pipeline = await get_pipeline()

    # Each WebSocket connection gets its own LangGraph session
    from src.api.web_pipeline import WebVoicePipeline

    try:
        conn_pipeline = WebVoicePipeline(
            vad=pipeline.vad,
            asr=pipeline.asr,
            tts_models=pipeline.tts_models,
        )
        thread_id = conn_pipeline.init_session()
        response_task = None

        await ws.send_json({
            "type": "session",
            "threadId": thread_id,
            "agent": conn_pipeline.active_agent_name,
        })

        async def send_callback(msg):
            """Send pipeline results back to the browser."""
            if msg["type"] == "audio":
                # Send audio as base64-encoded JSON so we can include metadata
                audio_b64 = base64.b64encode(msg["data"]).decode("ascii")
                await ws.send_json({
                    "type": "audio",
                    "data": audio_b64,
                    "sampleRate": msg["sample_rate"],
                })
            else:
                await ws.send_json(msg)

        async def cancel_response():
            nonlocal response_task
            if response_task is None:
                return

            if not response_task.done():
                conn_pipeline.cancel_active_response()
                response_task.cancel()
                with suppress(asyncio.CancelledError):
                    await response_task
            response_task = None

        async def start_response(user_text: str):
            nonlocal response_task
            await cancel_response()
            response_task = asyncio.create_task(
                conn_pipeline.generate_response(user_text, send_callback)
            )

        while True:
            data = await ws.receive()

            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"]:
                # Binary audio chunk from the browser mic
                raw_bytes = data["bytes"]
                result = conn_pipeline.process_audio_chunk(raw_bytes)

                if result["status"] == "speech_detected":
                    await cancel_response()
                    await ws.send_json({"type": "status", "value": "recording"})

                elif result["status"] == "speech_partial":
                    # Stream intermediate ASR transcript back to the UI
                    text = await asyncio.get_event_loop().run_in_executor(
                        None, conn_pipeline.transcribe, result["audio"]
                    )
                    if text and len(text.strip()) > 0:
                        await ws.send_json({
                            "type": "transcript",
                            "role": "user",
                            "text": text,
                            "partial": True
                        })

                elif result["status"] == "speech_end":
                    await ws.send_json({"type": "status", "value": "processing"})
                    text = await asyncio.get_event_loop().run_in_executor(
                        None, conn_pipeline.transcribe, result["audio"]
                    )
                    if text and len(text.strip()) >= 2:
                        # Send the final locked-in transcript
                        await ws.send_json({
                            "type": "transcript",
                            "role": "user",
                            "text": text,
                            "partial": False
                        })
                        # Generate the AI response using the final text
                        await start_response(text)
                    else:
                        await ws.send_json({"type": "status", "value": "idle"})

            elif "text" in data and data["text"]:
                # JSON text message (for text-chat mode)
                msg = json.loads(data["text"])
                if msg.get("type") == "text":
                    user_text = msg.get("text", "").strip()
                    if user_text:
                        await ws.send_json({
                            "type": "transcript",
                            "role": "user",
                            "text": user_text,
                            "partial": False,
                        })
                        await start_response(user_text)
                elif msg.get("type") == "stop_audio":
                    result = conn_pipeline.finalize_current_utterance()
                    if result["status"] != "speech_end":
                        await ws.send_json({"type": "status", "value": "idle"})
                        continue

                    await ws.send_json({"type": "status", "value": "processing"})
                    text = await asyncio.get_event_loop().run_in_executor(
                        None, conn_pipeline.transcribe, result["audio"]
                    )
                    if text and len(text.strip()) >= 2:
                        await ws.send_json({
                            "type": "transcript",
                            "role": "user",
                            "text": text,
                            "partial": False,
                        })
                        await start_response(text)
                    else:
                        await ws.send_json({"type": "status", "value": "idle"})

    except WebSocketDisconnect:
        print(f"[Server] Client disconnected globally")
    except Exception as e:
        print(f"[Server] WebSocket error: {e}")
        try:
            await ws.close()
        except:
            pass
    finally:
        with suppress(Exception):
            await cancel_response()


# ─── WebSocket: Text Chat Mode ─────────────────────────────────────────────

@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    pipeline = await get_pipeline()

    from src.api.web_pipeline import WebVoicePipeline

    try:
        conn_pipeline = WebVoicePipeline(
            vad=pipeline.vad,
            asr=pipeline.asr,
            tts_models=pipeline.tts_models,
        )
        thread_id = conn_pipeline.init_session()

        await ws.send_json({
            "type": "session",
            "threadId": thread_id,
            "agent": conn_pipeline.active_agent_name,
        })

        async def send_callback(msg):
            if msg["type"] == "audio":
                audio_b64 = base64.b64encode(msg["data"]).decode("ascii")
                await ws.send_json({
                    "type": "audio",
                    "data": audio_b64,
                    "sampleRate": msg["sample_rate"],
                })
            else:
                await ws.send_json(msg)

        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            user_text = msg.get("text", "").strip()
            if user_text:
                await ws.send_json({
                    "type": "transcript",
                    "role": "user",
                    "text": user_text,
                    "partial": False,
                })
                await conn_pipeline.generate_response(user_text, send_callback)
    except WebSocketDisconnect:
        print(f"[Server] Chat client disconnected globally")
    except Exception as e:
        print(f"[Server] Chat WebSocket error: {e}")
        try:
            await ws.close()
        except:
            pass
