import { useState, useCallback, useRef, useEffect } from 'react';
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
  const isAudioPlayingRef = useRef(false);

  const { isPlaying, startCapture, stopCapture, playAudioChunk, clearPlaybackQueue } = useAudio();

  // Track playing state to update pipeline status
  useEffect(() => {
    isAudioPlayingRef.current = isPlaying;
    if (isPlaying) {
      setPipelineStatus('speaking');
    }
  }, [isPlaying]);

  const handleMessage = useCallback((msg) => {
    switch (msg.type) {
      case 'session':
        setThreadId(msg.threadId);
        if (msg.agent) setActiveAgent(msg.agent);
        break;

      case 'agent':
        setActiveAgent(msg.name);
        break;

      case 'status':
        if (msg.value === 'idle' && !isAudioPlayingRef.current) {
          setPipelineStatus('idle');
        } else if (msg.value !== 'idle') {
          setPipelineStatus(msg.value);
        }
        break;

      case 'transcript':
        if (msg.role === 'user') {
          setMessages((prev) => [...prev, { role: 'user', text: msg.text }]);
          currentAiMessageRef.current = '';
        } else if (msg.role === 'ai') {
          if (msg.partial) {
            // Accumulate partial AI responses
            currentAiMessageRef.current += msg.text + ' ';
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
  }, [playAudioChunk]);

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
      setPipelineStatus('idle');
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
  }, [pipelineStatus, startCapture, stopCapture, sendBinary, clearPlaybackQueue]);

  /** Send a text message (text chat mode). */
  const sendTextMessage = useCallback((text) => {
    if (!text.trim()) return;
    // Add user message to chat immediately
    setMessages((prev) => [...prev, { role: 'user', text: text.trim() }]);
    sendJson({ type: 'text', text: text.trim() });
    currentAiMessageRef.current = '';
  }, [sendJson]);

  /** Switch between voice and text mode. */
  const switchMode = useCallback((newMode) => {
    stopCapture();
    clearPlaybackQueue();
    setPipelineStatus('idle');
    setMode(newMode);
  }, [stopCapture, clearPlaybackQueue]);

  return {
    // State
    activeAgent,
    pipelineStatus,
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
