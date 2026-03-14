# Big Win Combo Plan

## Goal

Turn OpenVoice AI into a genuinely low-latency realtime voice system by upgrading:

- ASR: from repeated offline-style chunk transcription to true streaming Qwen3-ASR
- Transport: from browser WebSocket audio to WebRTC for voice mode
- TTS: from sentence-first playback to lower-latency incremental playback
- Orchestration: from best-effort turn processing to explicit realtime session control
- Tooling: from demo tools to production-grade agent actions

## What The Internet Check Confirmed

From the official Qwen model card and toolkit docs:

- Qwen3-ASR-0.6B and 1.7B support both offline and streaming inference.
- Streaming inference is currently available only with the vLLM backend.
- The `qwen-asr` package supports both a transformers backend and a vLLM backend.
- Qwen recommends the vLLM backend for fastest inference.
- Streaming mode does not support batch inference or returning timestamps.
- Timestamps can be added through `Qwen3-ForcedAligner-0.6B`, but that is a separate alignment path.

## What The Repo Does Today

Current usage is not taking advantage of Qwen’s streaming path:

- [backend/src/asr/whisper.py](/home/badrinathan/Desktop/Projects/OpenVoice%20AI/backend/src/asr/whisper.py) loads `Qwen3ASRModel.from_pretrained(...)` using the default transformers-style backend.
- [backend/src/api/web_pipeline.py](/home/badrinathan/Desktop/Projects/OpenVoice%20AI/backend/src/api/web_pipeline.py) repeatedly calls `self.asr.transcribe(audio_array)` on growing partial buffers.
- [backend/src/api/server.py](/home/badrinathan/Desktop/Projects/OpenVoice%20AI/backend/src/api/server.py) sends microphone PCM chunks over WebSockets, but ASR remains request/response oriented.

So yes: the model supports streaming, but this codebase is currently treating it more like an offline transcriber that is called over and over on progressively larger windows.

## Big Win Combo: Recommended Execution Order

We should not jump straight to WebRTC first.

Recommended order:

1. Add latency tracing and turn-state instrumentation.
2. Replace current ASR path with true streaming Qwen3-ASR over vLLM.
3. Improve response kickoff and TTS chunking.
4. Migrate voice transport from WebSocket audio to WebRTC.
5. Add production-grade tool integrations and memory upgrades.

That order keeps the project shippable at every stage and prevents us from debugging transport and model orchestration at the same time.

## Phase 0: Baseline And Measurement

### Purpose

Before refactoring, capture where the current latency actually goes.

### Deliverables

- per-turn metrics:
  - `vad_start_ms`
  - `partial_asr_ms`
  - `final_asr_ms`
  - `llm_ttft_ms`
  - `tts_first_audio_ms`
  - `turn_total_ms`
  - `interrupt_cancel_ms`
- structured logs for one session ID / thread ID
- simple benchmark script for repeated spoken turns

### Why First

Without this, every “latency improvement” will be guesswork.

## Phase 1: True Streaming Qwen3-ASR With vLLM

### Purpose

Use the model the way Qwen documents it for realtime streaming instead of retranscribing growing buffers.

### Architecture Change

- Create a dedicated ASR service layer:
  - `backend/src/asr/streaming_client.py`
  - talks to a local `qwen-asr-serve` or `vllm serve` compatible endpoint
- Keep current `ASRModel` as fallback offline mode
- Add a new streaming session abstraction:
  - open stream
  - push PCM chunks
  - receive partial transcripts
  - receive final transcript
  - reset / cancel stream on barge-in

### Implementation Notes

- Keep current WebSocket browser transport for this phase.
- Replace `process_audio_chunk -> accumulate -> transcribe(full_buffer)` with:
  - VAD decides whether to forward chunks
  - forwarded chunks go immediately to ASR stream
  - ASR stream emits partials and finals
- Remove repeated full-buffer partial transcription in [backend/src/api/web_pipeline.py](/home/badrinathan/Desktop/Projects/OpenVoice%20AI/backend/src/api/web_pipeline.py)
- Add a fallback mode:
  - if vLLM ASR service is unavailable, use current offline backend

### Risks

- vLLM operational complexity
- GPU memory sizing
- partial transcript stability differences

### Success Criteria

- partial transcripts arrive continuously from Qwen’s streaming backend
- no repeated full-buffer retranscription
- lower `time-to-first-transcript`

## Phase 2: Faster Response Kickoff + Lower-Latency TTS

### Purpose

Start speaking sooner after partial/final user intent becomes clear.

### Deliverables

- early response kickoff policy:
  - trigger on confident final ASR
  - optional speculative kickoff on stable partials
- TTS chunking by clause or short phrase, not only full sentence boundaries
- explicit stale-audio suppression on cancel
- improved queue management in [backend/src/api/web_pipeline.py](/home/badrinathan/Desktop/Projects/OpenVoice%20AI/backend/src/api/web_pipeline.py)

### Optional Upgrade Paths

- keep Piper but stream smaller chunks sooner
- or evaluate a TTS engine with better realtime chunk generation if Piper is still the bottleneck

### Success Criteria

- lower `tts_first_audio_ms`
- reduced “thinking silence” before voice playback starts

## Phase 3: WebSocket Voice To WebRTC Voice

### Purpose

Reduce transport overhead and make browser audio truly realtime.

### Why Not First

If ASR is still offline-style, WebRTC alone will not solve the core latency problem.

### Architecture Change

- Keep WebSockets for:
  - control messages
  - chat mode fallback
  - server events if needed
- Use WebRTC for:
  - microphone uplink
  - AI audio downlink

### Deliverables

- FastAPI signaling endpoint for SDP / ICE exchange
- browser voice hook rewritten for WebRTC media tracks
- server-side RTC bridge component
- graceful fallback to WebSocket audio if WebRTC setup fails

### Benefits

- lower audio transport overhead
- better jitter handling
- cleaner realtime audio semantics
- easier future echo cancellation improvements

### Risks

- NAT / ICE / TURN complexity
- more moving parts in deployment

### Success Criteria

- stable voice sessions over WebRTC
- reduced end-to-end latency under real browser usage

## Phase 4: Production Agent And Tooling Upgrade

### Purpose

Make the system more powerful, not just faster.

### Deliverables

- replace mock tools with real integrations:
  - product catalog search
  - order lookup
  - return policy retrieval
  - customer profile / CRM state
- user memory:
  - preferences
  - previous unresolved issues
  - recent order context
- small routing layer before the main agents for faster specialist selection

### Success Criteria

- the agent layer performs useful real actions
- fewer handoff mistakes
- better continuity across sessions

## Phase 5: Optional Advanced Wins

- real-time transcript stabilization model
- forced aligner as an async post-processing step for transcript timing
- adaptive model routing:
  - tiny model for routing
  - bigger model for complex replies
- multilingual voice switching
- sentiment-aware prosody selection

## Concrete Repo Plan

### Milestone A

Ship a measurable streaming-ASR backend while keeping the current frontend transport.

Work items:

- add ASR provider abstraction
- add vLLM-backed streaming ASR client
- update voice pipeline to consume streaming partial/final events
- add latency metrics and debug logs

### Milestone B

Reduce response latency after ASR.

Work items:

- tighten TTS chunking
- optimize cancellation path
- start LLM/TTS earlier once transcript confidence is sufficient

### Milestone C

Move browser voice transport to WebRTC.

Work items:

- add signaling endpoints
- build browser WebRTC voice hook
- support AI audio track playback from server
- keep WebSocket fallback

### Milestone D

Make the agents actually powerful.

Work items:

- connect real tools
- add persistent user memory
- add routing improvements

## Recommended Immediate Start

Start with Milestone A only.

That gives the biggest latency win with the least product risk and is required before WebRTC is worth the migration cost.

## Source Notes

Primary sources used:

- Official Qwen model card:
  - https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- Official toolkit repo notice:
  - https://github.com/QwenLM/Qwen3-ASR-Toolkit

Key source takeaways:

- Qwen3-ASR supports offline and streaming inference.
- Streaming is currently available only with the vLLM backend.
- vLLM is the recommended fastest backend.
- Streaming mode does not return timestamps.
