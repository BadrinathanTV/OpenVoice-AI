# Frontend

React 19 + Vite frontend for the OpenVoice AI browser client.

## What It Does

- Opens a WebSocket connection to the backend voice or chat endpoint
- Streams microphone PCM audio in voice mode
- Renders live partial/final transcripts
- Plays streamed TTS audio from the backend
- Shows the active agent and pipeline state

## Commands

```bash
npm install
npm run dev
```

For a production-style preview:

```bash
npm run build
npm run preview -- --host 0.0.0.0 --port 5173
```

## Backend Connection

By default the frontend talks to port `8000` on the current host.

Optional environment variables:

```bash
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
```

If unset, the app derives `http/ws` vs `https/wss` from the current page.

## Browser Notes

- Microphone access is required for voice mode
- WebSocket audio streaming assumes 16 kHz mono PCM chunks
- The UI expects backend messages for:
  - `session`
  - `status`
  - `agent`
  - `transcript`
  - `audio`
