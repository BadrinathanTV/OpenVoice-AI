import { useState, useRef, useCallback, useEffect } from 'react';

/**
 * Custom hook for browser audio capture and playback via the Web Audio API.
 * Follows SRP: only handles audio I/O, no WebSocket or business logic.
 *
 * Captures raw 16-bit PCM at 16kHz mono for compatibility with the backend.
 */
export function useAudio() {
  const [isCapturing, setIsCapturing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const workletNodeRef = useRef(null);
  const playbackQueueRef = useRef([]);
  const isPlayingRef = useRef(false);
  const currentSourceRef = useRef(null);
  const processPlaybackQueueRef = useRef(null);

  /** Get or create the AudioContext (lazy init to avoid autoplay policy issues). */
  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000,
      });
    }
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }
    return audioContextRef.current;
  }, []);

  /**
   * Start capturing mic audio.
   * @param {function} onChunk - Called with Int16Array PCM chunks (~32ms each)
   */
  const startCapture = useCallback(async (onChunk) => {
    try {
      const ctx = getAudioContext();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      mediaStreamRef.current = stream;

      // Use ScriptProcessorNode as a simpler fallback (AudioWorklet requires HTTPS)
      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(512, 1, 1);
      const sink = ctx.createGain();
      sink.gain.value = 0;

      processor.onaudioprocess = (e) => {
        const float32 = e.inputBuffer.getChannelData(0);
        // Convert float32 → int16 PCM
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
          const s = Math.max(-1, Math.min(1, float32[i]));
          int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        onChunk(int16.buffer);
      };

      source.connect(processor);
      processor.connect(sink); // ScriptProcessor must stay connected to process audio.
      sink.connect(ctx.destination);
      workletNodeRef.current = { source, processor, sink };
      setIsCapturing(true);
    } catch (err) {
      console.error('[Audio] Mic capture failed:', err);
      throw err;
    }
  }, [getAudioContext]);

  /** Stop mic capture. */
  const stopCapture = useCallback(() => {
    if (workletNodeRef.current) {
      const { source, processor, sink } = workletNodeRef.current;
      processor.disconnect();
      source.disconnect();
      sink.disconnect();
      workletNodeRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    setIsCapturing(false);
  }, []);

  const processPlaybackQueue = useCallback(() => {
    if (isPlayingRef.current || playbackQueueRef.current.length === 0) return;

    isPlayingRef.current = true;
    setIsPlaying(true);

    const { float32, sampleRate } = playbackQueueRef.current.shift();
    const ctx = getAudioContext();

    const buffer = ctx.createBuffer(1, float32.length, sampleRate);
    buffer.getChannelData(0).set(float32);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    currentSourceRef.current = source;

    source.onended = () => {
      if (currentSourceRef.current === source) {
        currentSourceRef.current = null;
      }
      isPlayingRef.current = false;
      if (playbackQueueRef.current.length > 0) {
        processPlaybackQueueRef.current?.();
      } else {
        setIsPlaying(false);
      }
    };

    source.start();
  }, [getAudioContext]);

  useEffect(() => {
    processPlaybackQueueRef.current = processPlaybackQueue;
  }, [processPlaybackQueue]);

  /**
   * Enqueue base64-encoded 16-bit PCM audio for playback.
   * Plays sequentially (FIFO) to avoid overlapping sentences.
   */
  const playAudioChunk = useCallback((base64Data, sampleRate = 16000) => {
    const binaryStr = atob(base64Data);
    const bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }
    const int16 = new Int16Array(bytes.buffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768.0;
    }

    playbackQueueRef.current.push({ float32, sampleRate });
    processPlaybackQueue();
  }, [processPlaybackQueue]);

  /** Clear the playback queue (e.g. on interrupt). */
  const clearPlaybackQueue = useCallback(() => {
    playbackQueueRef.current = [];
    if (currentSourceRef.current) {
      try {
        currentSourceRef.current.onended = null;
        currentSourceRef.current.stop();
      } catch {
        // Ignore already-stopped playback.
      }
      currentSourceRef.current = null;
    }
    isPlayingRef.current = false;
    setIsPlaying(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCapture();
      clearPlaybackQueue();
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
    };
  }, [stopCapture, clearPlaybackQueue]);

  return {
    isCapturing,
    isPlaying,
    startCapture,
    stopCapture,
    playAudioChunk,
    clearPlaybackQueue,
  };
}
