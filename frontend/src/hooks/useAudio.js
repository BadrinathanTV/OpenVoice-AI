import { useState, useRef, useCallback, useEffect } from 'react';

export function useAudio() {
  const [isCapturing, setIsCapturing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isUserSpeaking, setIsUserSpeaking] = useState(false);

  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const workletNodeRef = useRef(null);
  const playbackQueueRef = useRef([]);
  const isPlayingRef = useRef(false);
  const currentSourceRef = useRef(null);
  const processPlaybackQueueRef = useRef(null);
  const speechReleaseTimeoutRef = useRef(null);

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

      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(512, 1, 1);
      const sink = ctx.createGain();
      sink.gain.value = 0;

      processor.onaudioprocess = (event) => {
        const float32 = event.inputBuffer.getChannelData(0);
        const int16 = new Int16Array(float32.length);
        let sumSquares = 0;

        for (let i = 0; i < float32.length; i += 1) {
          const sample = Math.max(-1, Math.min(1, float32[i]));
          sumSquares += sample * sample;
          int16[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
        }

        const rms = Math.sqrt(sumSquares / float32.length);
        const speechThreshold = 0.028;

        if (rms > speechThreshold) {
          if (speechReleaseTimeoutRef.current) {
            clearTimeout(speechReleaseTimeoutRef.current);
            speechReleaseTimeoutRef.current = null;
          }
          setIsUserSpeaking(true);
        } else if (!speechReleaseTimeoutRef.current) {
          speechReleaseTimeoutRef.current = setTimeout(() => {
            setIsUserSpeaking(false);
            speechReleaseTimeoutRef.current = null;
          }, 140);
        }

        onChunk(int16.buffer);
      };

      source.connect(processor);
      processor.connect(sink);
      sink.connect(ctx.destination);
      workletNodeRef.current = { source, processor, sink };
      setIsCapturing(true);
    } catch (err) {
      console.error('[Audio] Mic capture failed:', err);
      throw err;
    }
  }, [getAudioContext]);

  const stopCapture = useCallback(() => {
    if (workletNodeRef.current) {
      const { source, processor, sink } = workletNodeRef.current;
      processor.disconnect();
      source.disconnect();
      sink.disconnect();
      workletNodeRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (speechReleaseTimeoutRef.current) {
      clearTimeout(speechReleaseTimeoutRef.current);
      speechReleaseTimeoutRef.current = null;
    }
    setIsUserSpeaking(false);
    setIsCapturing(false);
  }, []);

  /**
   * Temporarily mute mic capture by disconnecting the processor.
   * The mic stream stays open so resume is instant (no getUserMedia delay).
   * This prevents the AI's playback audio from leaking back through
   * speakers → mic → backend, which is the root cause of double-voice.
   */
  const muteCapture = useCallback(() => {
    if (workletNodeRef.current) {
      const { source, processor } = workletNodeRef.current;
      try {
        source.disconnect(processor);
      } catch {
        // Already disconnected — safe to ignore.
      }
    }
  }, []);

  /**
   * Resume mic capture after AI playback finishes.
   */
  const unmuteCapture = useCallback(() => {
    if (workletNodeRef.current) {
      const { source, processor } = workletNodeRef.current;
      try {
        source.connect(processor);
      } catch {
        // Already connected — safe to ignore.
      }
    }
  }, []);

  /**
   * Gapless audio playback: schedule chunks at precise future timestamps
   * so there are no micro-gaps between sequential audio segments.
   * Track ALL active sources so barge-in can stop them all.
   */
  const nextPlaybackTimeRef = useRef(0);
  const activeSourcesRef = useRef(new Set());

  const processPlaybackQueue = useCallback(() => {
    if (playbackQueueRef.current.length === 0) {
      if (!isPlayingRef.current) return;
      return;
    }

    const ctx = getAudioContext();
    const now = ctx.currentTime;

    if (!isPlayingRef.current || nextPlaybackTimeRef.current < now) {
      nextPlaybackTimeRef.current = now;
    }
    
    isPlayingRef.current = true;
    setIsPlaying(true);

    const { float32, sampleRate } = playbackQueueRef.current.shift();
    const buffer = ctx.createBuffer(1, float32.length, sampleRate);
    buffer.getChannelData(0).set(float32);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    currentSourceRef.current = source;
    activeSourcesRef.current.add(source);

    const startAt = nextPlaybackTimeRef.current;
    const duration = float32.length / sampleRate;
    nextPlaybackTimeRef.current = startAt + duration;

    source.onended = () => {
      activeSourcesRef.current.delete(source);
      if (currentSourceRef.current === source) {
        currentSourceRef.current = null;
      }
      if (playbackQueueRef.current.length > 0) {
        processPlaybackQueueRef.current?.();
      } else if (activeSourcesRef.current.size === 0) {
        isPlayingRef.current = false;
        setIsPlaying(false);
      }
    };

    source.start(startAt);

    if (playbackQueueRef.current.length > 0) {
      processPlaybackQueueRef.current?.();
    }
  }, [getAudioContext]);

  useEffect(() => {
    processPlaybackQueueRef.current = processPlaybackQueue;
  }, [processPlaybackQueue]);

  const playAudioChunk = useCallback((base64Data, sampleRate = 16000) => {
    const binaryStr = atob(base64Data);
    const bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i += 1) {
      bytes[i] = binaryStr.charCodeAt(i);
    }

    const int16 = new Int16Array(bytes.buffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i += 1) {
      float32[i] = int16[i] / 32768.0;
    }

    playbackQueueRef.current.push({ float32, sampleRate });
    processPlaybackQueue();
  }, [processPlaybackQueue]);

  const clearPlaybackQueue = useCallback(() => {
    playbackQueueRef.current = [];
    nextPlaybackTimeRef.current = 0;
    // Stop ALL pre-scheduled sources, not just the last one
    for (const src of activeSourcesRef.current) {
      try {
        src.onended = null;
        src.stop();
      } catch {
        // Already stopped
      }
    }
    activeSourcesRef.current.clear();
    currentSourceRef.current = null;
    isPlayingRef.current = false;
    setIsPlaying(false);
  }, []);

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
    isUserSpeaking,
    startCapture,
    stopCapture,
    muteCapture,
    unmuteCapture,
    playAudioChunk,
    clearPlaybackQueue,
  };
}
