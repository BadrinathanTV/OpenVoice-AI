import { useState, useCallback, useEffect, useRef } from 'react';
import { useWebSocket } from './useWebSocket';
import { useAudio } from './useAudio';
import { DEFAULT_AGENT } from '../config/agents';

/**
 * Orchestration hook — coordinates WebSocket, audio, and UI state.
 * Follows the Interface Segregation Principle: exposes a clean API to components
 * without leaking implementation details of WS or Audio internals.
 */
export function useVoicePipeline() {
  const [activeAgent, setActiveAgent] = useState(DEFAULT_AGENT);
  const [pipelineStatus, setPipelineStatus] = useState('idle'); // idle, recording, processing, thinking, speaking
  const [messages, setMessages] = useState([]);
  const [threadId, setThreadId] = useState(null);
  const [mode, setMode] = useState('voice'); // 'voice' | 'text'

  const currentAiMessageRef = useRef('');
  const currentUserMessageRef = useRef('');
  const isAudioPlayingRef = useRef(false);

  const { isPlaying, startCapture, stopCapture, playAudioChunk, clearPlaybackQueue } = useAudio();

  useEffect(() => {
    isAudioPlayingRef.current = isPlaying;
  }, [isPlaying]);

  const commitPendingAiMessage = useCallback(() => {
    setMessages((prev) => {
      const copy = [...prev];
      const last = copy[copy.length - 1];
      if (last && last.role === 'ai' && last.partial) {
        copy[copy.length - 1] = {
          ...last,
          partial: false,
        };
      }
      return copy;
    });
    currentAiMessageRef.current = '';
  }, []);

  const handleMessage = useCallback((msg) => {
    switch (msg.type) {
      case 'session':
        setThreadId((prevThreadId) => {
          if (prevThreadId && prevThreadId !== msg.threadId) {
            setMessages([]);
            currentAiMessageRef.current = '';
            currentUserMessageRef.current = '';
            setPipelineStatus('idle');
          }
          return msg.threadId;
        });
        if (msg.agent) setActiveAgent(msg.agent);
        break;

      case 'agent':
        setActiveAgent(msg.name);
        break;

      case 'status':
        // Barge-in: if user starts speaking while AI audio is playing, interrupt it
        if (msg.value === 'recording' && isAudioPlayingRef.current) {
          clearPlaybackQueue();
        }
        if (msg.value === 'recording' || msg.value === 'processing') {
          commitPendingAiMessage();
        }
        if (msg.value === 'idle' && isAudioPlayingRef.current) {
          // Don't override 'speaking' status while audio is still playing
          break;
        }
        setPipelineStatus(msg.value);
        break;

      case 'transcript':
        if (msg.role === 'user') {
          if (msg.partial) {
            currentUserMessageRef.current = msg.text;
            setMessages((prev) => {
              const copy = [...prev];
              const last = copy[copy.length - 1];
              if (last && last.role === 'user' && last.partial) {
                copy[copy.length - 1] = {
                  ...last,
                  text: currentUserMessageRef.current,
                };
              } else {
                copy.push({
                  role: 'user',
                  text: currentUserMessageRef.current,
                  partial: true,
                });
              }
              return copy;
            });
          } else {
            setMessages((prev) => {
              const copy = [...prev];
              const last = copy[copy.length - 1];
              if (last && last.role === 'user' && last.partial) {
                copy[copy.length - 1] = {
                  role: 'user',
                  text: msg.text,
                  partial: false,
                };
              } else {
                copy.push({
                  role: 'user',
                  text: msg.text,
                  partial: false,
                });
              }
              return copy;
            });
            currentUserMessageRef.current = '';
            commitPendingAiMessage();
          }
        } else if (msg.role === 'ai') {
          if (msg.partial) {
            // Accumulate partial AI responses without adding custom spaces,
            // as raw tokens directly from the LLM usually include appropriate spacing.
            currentAiMessageRef.current += msg.text;
            setMessages((prev) => {
              const copy = [...prev];
              const last = copy[copy.length - 1];
              if (last && last.role === 'ai' && last.partial) {
                // Update the in-progress AI message
                copy[copy.length - 1] = {
                  ...last,
                  text: currentAiMessageRef.current.trim(),
                  agent: msg.agent,
                };
              } else {
                copy.push({
                  role: 'ai',
                  text: currentAiMessageRef.current.trim(),
                  agent: msg.agent,
                  partial: true,
                });
              }
              return copy;
            });
          } else {
            // Final complete message
            setMessages((prev) => {
              const copy = [...prev];
              const last = copy[copy.length - 1];
              if (last && last.role === 'ai' && last.partial) {
                copy[copy.length - 1] = {
                  role: 'ai',
                  text: msg.text,
                  agent: msg.agent,
                  partial: false,
                };
              }
              return copy;
            });
            currentAiMessageRef.current = '';
          }
        }
        break;

      case 'audio':
        playAudioChunk(msg.data, msg.sampleRate);
        break;

      default:
        break;
    }
  }, [clearPlaybackQueue, commitPendingAiMessage, playAudioChunk]);

  const wsHandlers = {
    onMessage: handleMessage,
    onOpen: () => console.log('[Pipeline] WebSocket connected'),
    onClose: () => console.log('[Pipeline] WebSocket disconnected'),
  };

  const { status: wsStatus, sendJson, sendBinary } = useWebSocket(
    mode === 'voice' ? '/ws/voice' : '/ws/chat',
    wsHandlers,
    true
  );

  /** Toggle voice capture on/off. */
  const toggleVoice = useCallback(async () => {
    if (pipelineStatus === 'recording' || pipelineStatus === 'accumulating') {
      stopCapture();
      setPipelineStatus('processing');
      sendJson({ type: "stop_audio" });
    } else {
      clearPlaybackQueue();
      setPipelineStatus('recording');
      try {
        await startCapture((pcmBuffer) => {
          sendBinary(pcmBuffer);
        });
      } catch {
        setPipelineStatus('idle');
      }
    }
  }, [pipelineStatus, startCapture, stopCapture, sendJson, sendBinary, clearPlaybackQueue]);

  /** Send a text message (text chat mode). */
  const sendTextMessage = useCallback((text) => {
    if (!text.trim()) return;
    sendJson({ type: 'text', text: text.trim() });
    currentAiMessageRef.current = '';
  }, [sendJson]);

  /** Switch between voice and text mode. */
  const switchMode = useCallback((newMode) => {
    stopCapture();
    clearPlaybackQueue();
    setPipelineStatus('idle');
    setMessages([]);
    setThreadId(null);
    currentAiMessageRef.current = '';
    currentUserMessageRef.current = '';
    setMode(newMode);
  }, [stopCapture, clearPlaybackQueue]);

  const effectivePipelineStatus = isPlaying ? 'speaking' : pipelineStatus;

  return {
    // State
    activeAgent,
    pipelineStatus: effectivePipelineStatus,
    messages,
    threadId,
    mode,
    wsStatus,
    isPlaying,

    // Actions
    toggleVoice,
    sendTextMessage,
    switchMode,
  };
}
