"""
FastAPI server with realtime endpoints for the OpenVoice AI assistant.
Voice mode uses WebRTC transport, while text mode remains on WebSocket.
"""

import asyncio
import base64
import json
import logging
import os
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
except Exception:  # pragma: no cover - runtime dependency check
    RTCPeerConnection = None  # type: ignore[assignment]
    RTCSessionDescription = None  # type: ignore[assignment]

WEBRTC_VERBOSE = os.getenv("OPENVOICE_WEBRTC_VERBOSE", "0").lower() in {
    "1", "true", "yes", "on"
}
if not WEBRTC_VERBOSE:
    # Silence noisy per-candidate ICE logs by default.
    for logger_name in ("aioice", "aioice.ice", "aiortc", "httpx", "openai"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

if TYPE_CHECKING:
    from src.api.web_pipeline import WebVoicePipeline

# ─── Lazy-loaded heavy AI components ────────────────────────────────────────
# We delay imports of torch / ML models until they are actually needed so the
# server process starts fast and the /api/health endpoint works instantly.

_pipeline_instance: "WebVoicePipeline | None" = None
_pipeline_lock = asyncio.Lock()
_pipeline_init_error: str | None = None
_webrtc_peer_connections: dict[str, Any] = {}
SendCallback = Callable[[dict[str, Any]], Awaitable[None]]


class WebRTCOffer(BaseModel):
    sdp: str
    type: str


def _build_pipeline() -> "WebVoicePipeline":
    """Instantiate the heavy ML models (runs once, in-process)."""
    from src.vad.silero import SileroVAD
    from src.asr.qwen import ASRModel
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


def _format_pipeline_error(exc: Exception) -> str:
    message = str(exc)
    if "vLLM is not available" in message:
        return (
            "ASR backend is set to 'vllm', but vLLM is not installed in this uv "
            "environment. Run `uv sync --extra streaming-asr` from the repo root, "
            "or switch `ASR_BACKEND=transformers` in backend/.env."
        )
    return message


async def get_pipeline() -> "WebVoicePipeline":
    """Thread-safe singleton accessor for the pipeline."""
    global _pipeline_init_error, _pipeline_instance
    if _pipeline_init_error is not None:
        raise RuntimeError(_pipeline_init_error)

    if _pipeline_instance is None:
        async with _pipeline_lock:
            if _pipeline_init_error is not None:
                raise RuntimeError(_pipeline_init_error)
            if _pipeline_instance is None:
                try:
                    _pipeline_instance = await asyncio.get_event_loop().run_in_executor(
                        None, _build_pipeline
                    )
                    _pipeline_init_error = None
                except Exception as exc:
                    _pipeline_init_error = _format_pipeline_error(exc)
                    raise RuntimeError(_pipeline_init_error) from exc
    assert _pipeline_instance is not None
    return _pipeline_instance


async def _build_connection_pipeline() -> "WebVoicePipeline":
    pipeline = await get_pipeline()
    from src.api.web_pipeline import WebVoicePipeline

    return WebVoicePipeline(
        vad=pipeline.vad,
        asr=pipeline.asr,
        tts_models=pipeline.tts_models,
    )


async def _wait_for_ice_complete(peer_connection: Any) -> None:
    if peer_connection.iceGatheringState == "complete":
        return

    ice_complete = asyncio.Event()

    @peer_connection.on("icegatheringstatechange")
    async def _on_ice_state_change() -> None:
        if peer_connection.iceGatheringState == "complete":
            ice_complete.set()

    await ice_complete.wait()


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


@app.on_event("shutdown")
async def close_webrtc_peers() -> None:
    if not _webrtc_peer_connections:
        return

    peers = list(_webrtc_peer_connections.values())
    _webrtc_peer_connections.clear()
    for peer in peers:
        with suppress(Exception):
            await peer.close()


# ─── REST Endpoints ────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok" if _pipeline_init_error is None else "error",
        "models_loaded": _pipeline_instance is not None,
        "init_error": _pipeline_init_error,
    }


@app.get("/api/agents")
async def list_agents():
    return {
        "agents": [
            {"name": "CustomerCare", "description": "General help, returns, refunds, policies", "color": "#6C63FF"},
            {"name": "Shopper", "description": "Product search and recommendations", "color": "#00C9A7"},
            {"name": "OrderOps", "description": "Order tracking and operations", "color": "#FF6B6B"},
        ]
    }


# ─── WebRTC: Voice Mode ────────────────────────────────────────────────────

@app.post("/api/webrtc/offer")
async def webrtc_offer(offer: WebRTCOffer):
    if RTCPeerConnection is None or RTCSessionDescription is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "WebRTC transport requires aiortc. "
                "Install dependencies with `uv sync`."
            ),
        )

    conn_pipeline = await _build_connection_pipeline()
    session_id = f"rtc_{uuid.uuid4().hex[:8]}"
    print(f"[VoiceSession] Initializing swarm for {session_id}...")
    thread_id = conn_pipeline.init_session()
    peer_connection = RTCPeerConnection()
    _webrtc_peer_connections[session_id] = peer_connection
    print(f"[Transport] WebRTC voice session created: {session_id}")

    response_task: asyncio.Task[None] | None = None
    response_seq = 0
    active_response_id: int | None = None
    control_channel: Any = None
    transport_closed = False
    cleanup_started = False

    async def send_control(payload: dict[str, Any]) -> None:
        channel = control_channel
        if (
            transport_closed
            or channel is None
            or getattr(channel, "readyState", None) != "open"
        ):
            return
        channel.send(json.dumps(payload))

    def make_send_callback(response_id: int) -> SendCallback:
        async def send_callback(msg: dict[str, Any]) -> None:
            if transport_closed or active_response_id != response_id:
                return

            if msg["type"] == "audio":
                audio_b64 = base64.b64encode(msg["data"]).decode("ascii")
                await send_control({
                    "type": "audio",
                    "data": audio_b64,
                    "sampleRate": msg["sample_rate"],
                })
                return

            await send_control(msg)

        return send_callback

    async def cancel_response() -> None:
        nonlocal active_response_id, response_task
        active_response_id = None
        if response_task is None:
            return

        if not response_task.done():
            conn_pipeline.cancel_active_response()
            response_task.cancel()
            with suppress(asyncio.CancelledError):
                await response_task
        response_task = None

    async def start_response(user_text: str) -> None:
        nonlocal active_response_id, response_seq, response_task
        await cancel_response()
        response_seq += 1
        active_response_id = response_seq
        response_task = asyncio.create_task(
            conn_pipeline.generate_response(
                user_text,
                make_send_callback(active_response_id),
            )
        )

    async def handle_audio_chunk(raw_bytes: bytes) -> None:
        result = conn_pipeline.process_audio_chunk(raw_bytes)

        if result["status"] == "speech_detected":
            await cancel_response()
            await send_control({"type": "status", "value": "recording"})
            return

        if result["status"] == "speech_partial":
            if "text" in result:
                text = str(result["text"]).strip()
            else:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, conn_pipeline.transcribe, result["audio"]
                )
            if text:
                await send_control({
                    "type": "transcript",
                    "role": "user",
                    "text": text,
                    "partial": True,
                })
            return

        if result["status"] != "speech_end":
            return

        await send_control({"type": "status", "value": "processing"})
        if "text" in result:
            text = str(result["text"]).strip()
        else:
            text = await asyncio.get_event_loop().run_in_executor(
                None, conn_pipeline.transcribe, result["audio"]
            )
        if text and len(text.strip()) >= 2:
            await send_control({
                "type": "transcript",
                "role": "user",
                "text": text,
                "partial": False,
            })
            await start_response(text)
            return

        await send_control({"type": "status", "value": "idle"})

    async def handle_control_message(raw_text: str) -> None:
        try:
            msg = json.loads(raw_text)
        except json.JSONDecodeError:
            return

        if msg.get("type") == "text":
            user_text = str(msg.get("text", "")).strip()
            if user_text:
                await send_control({
                    "type": "transcript",
                    "role": "user",
                    "text": user_text,
                    "partial": False,
                })
                await start_response(user_text)
            return

        if msg.get("type") == "toggle_denoising":
            conn_pipeline.use_denoising = bool(msg.get("enabled", False))
            return

        if msg.get("type") != "stop_audio":
            return

        result = conn_pipeline.finalize_current_utterance()
        if result["status"] != "speech_end":
            await send_control({"type": "status", "value": "idle"})
            return

        await send_control({"type": "status", "value": "processing"})
        if "text" in result:
            text = str(result["text"]).strip()
        else:
            text = await asyncio.get_event_loop().run_in_executor(
                None, conn_pipeline.transcribe, result["audio"]
            )
        if text and len(text.strip()) >= 2:
            await send_control({
                "type": "transcript",
                "role": "user",
                "text": text,
                "partial": False,
            })
            await start_response(text)
            return

        await send_control({"type": "status", "value": "idle"})

    async def cleanup(close_peer: bool = True) -> None:
        nonlocal cleanup_started, transport_closed
        if cleanup_started:
            return

        cleanup_started = True
        transport_closed = True
        await cancel_response()
        _webrtc_peer_connections.pop(session_id, None)
        if close_peer:
            with suppress(Exception):
                await peer_connection.close()
        print(f"[Transport] WebRTC voice session closed: {session_id}")

    @peer_connection.on("datachannel")
    def on_datachannel(channel: Any) -> None:
        nonlocal control_channel

        if channel.label == "ov-control":
            control_channel = channel

            @channel.on("open")
            def on_control_open() -> None:
                print(f"[Transport] WebRTC control channel open: {session_id}")
                asyncio.create_task(
                    send_control({
                        "type": "session",
                        "threadId": thread_id,
                        "agent": conn_pipeline.active_agent_name,
                    })
                )

            @channel.on("message")
            def on_control_message(message: Any) -> None:
                if isinstance(message, str):
                    asyncio.create_task(handle_control_message(message))

            @channel.on("close")
            def on_control_close() -> None:
                asyncio.create_task(cleanup())

            return

        if channel.label != "ov-audio-up":
            return

        @channel.on("message")
        def on_audio_message(message: Any) -> None:
            if isinstance(message, (bytes, bytearray)):
                asyncio.create_task(handle_audio_chunk(bytes(message)))

    @peer_connection.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        state = peer_connection.connectionState
        if state in {"failed", "disconnected", "closed"}:
            await cleanup(close_peer=state != "closed")

    try:
        await peer_connection.setRemoteDescription(
            RTCSessionDescription(sdp=offer.sdp, type=offer.type)
        )
        answer = await peer_connection.createAnswer()
        await peer_connection.setLocalDescription(answer)
        await _wait_for_ice_complete(peer_connection)
    except Exception as exc:
        await cleanup()
        raise HTTPException(status_code=500, detail=f"WebRTC setup failed: {exc}") from exc

    local_description = peer_connection.localDescription
    if local_description is None:
        await cleanup()
        raise HTTPException(status_code=500, detail="WebRTC local description missing.")

    return {
        "sdp": local_description.sdp,
        "type": local_description.type,
        "sessionId": session_id,
    }


# ─── WebSocket: Voice Mode ─────────────────────────────────────────────────

@app.websocket("/ws/voice")
async def websocket_voice(ws: WebSocket):
    await ws.accept()
    client_host = ws.client.host if ws.client else "unknown"
    print(f"[Transport] Voice channel connected from {client_host}")
    try:
        pipeline = await get_pipeline()
    except RuntimeError as exc:
        await ws.send_json({"type": "error", "message": str(exc)})
        await ws.close(code=1011, reason="Backend initialization failed")
        return

    # Each WebSocket connection gets its own LangGraph session
    from src.api.web_pipeline import WebVoicePipeline

    try:
        conn_pipeline = WebVoicePipeline(
            vad=pipeline.vad,
            asr=pipeline.asr,
            tts_models=pipeline.tts_models,
        )
        thread_id = conn_pipeline.init_session()
        response_task: asyncio.Task[None] | None = None
        response_seq = 0
        active_response_id: int | None = None
        ws_closed = False

        await ws.send_json({
            "type": "session",
            "threadId": thread_id,
            "agent": conn_pipeline.active_agent_name,
        })

        def make_send_callback(response_id: int) -> SendCallback:
            async def send_callback(msg: dict[str, Any]) -> None:
                """Send only the current response stream back to the browser."""
                if ws_closed or active_response_id != response_id:
                    return

                if msg["type"] == "audio":
                    audio_b64 = base64.b64encode(msg["data"]).decode("ascii")
                    await ws.send_json({
                        "type": "audio",
                        "data": audio_b64,
                        "sampleRate": msg["sample_rate"],
                    })
                else:
                    await ws.send_json(msg)

            return send_callback

        async def cancel_response() -> None:
            nonlocal active_response_id, response_task
            active_response_id = None
            if response_task is None:
                return

            if not response_task.done():
                conn_pipeline.cancel_active_response()
                response_task.cancel()
                with suppress(asyncio.CancelledError):
                    await response_task
            response_task = None

        async def start_response(user_text: str) -> None:
            nonlocal active_response_id, response_seq, response_task
            await cancel_response()
            response_seq += 1
            active_response_id = response_seq
            response_task = asyncio.create_task(
                conn_pipeline.generate_response(
                    user_text,
                    make_send_callback(active_response_id),
                )
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
                    if "text" in result:
                        text = str(result["text"]).strip()
                    else:
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
                    if "text" in result:
                        text = str(result["text"]).strip()
                    else:
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
                    if "text" in result:
                        text = str(result["text"]).strip()
                    else:
                        text = await asyncio.get_event_loop().run_in_executor(
                            None, conn_pipeline.transcribe, result["audio"]
                        )
                    if text and len(text.strip()) >= 2:
                        await ws.send_json({
                            "type": "transcript",
                            "role": "user",
                            "text": text,
                            "partial": False
                        })
                        await start_response(text)
                    else:
                        await ws.send_json({"type": "status", "value": "idle"})
                    continue

                elif msg.get("type") == "toggle_denoising":
                    conn_pipeline.use_denoising = bool(msg.get("enabled", False))
                    continue

    except WebSocketDisconnect:
        ws_closed = True
        print(f"[Transport] Voice channel disconnected from {client_host}")
    except Exception as e:
        ws_closed = True
        print(f"[Transport] Voice channel error: {e}")
        try:
            await ws.close()
        except:
            pass
    finally:
        ws_closed = True
        with suppress(Exception):
            await cancel_response()


# ─── WebSocket: Text Chat Mode ─────────────────────────────────────────────

@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    client_host = ws.client.host if ws.client else "unknown"
    print(f"[Transport] Chat channel connected from {client_host}")
    try:
        pipeline = await get_pipeline()
    except RuntimeError as exc:
        await ws.send_json({"type": "error", "message": str(exc)})
        await ws.close(code=1011, reason="Backend initialization failed")
        return

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

        async def send_callback(msg: dict[str, Any]) -> None:
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
        print(f"[Transport] Chat channel disconnected from {client_host}")
    except Exception as e:
        print(f"[Transport] Chat channel error: {e}")
        try:
            await ws.close()
        except:
            pass
