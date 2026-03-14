import { useState, useCallback, useEffect, useRef } from 'react';
import { useWebSocket } from './useWebSocket';
import { useWebRTC } from './useWebRTC';
import { useAudio } from './useAudio';
import { DEFAULT_AGENT } from '../config/agents';

export function useVoicePipeline() {
  const [activeAgent, setActiveAgent] = useState(DEFAULT_AGENT);
  const [pipelineStatus, setPipelineStatus] = useState('idle');
  const [messages, setMessages] = useState([]);
  const [threadId, setThreadId] = useState(null);
  const [mode, setMode] = useState('voice');

  const currentAiMessageRef = useRef('');
  const currentUserMessageRef = useRef('');
  const isAudioPlayingRef = useRef(false);

  const {
    isPlaying,
    isUserSpeaking,
    startCapture,
    stopCapture,
    playAudioChunk,
    clearPlaybackQueue,
  } = useAudio();

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
        if (msg.value === 'recording' && isAudioPlayingRef.current) {
          clearPlaybackQueue();
        }
        if (msg.value === 'recording' || msg.value === 'processing') {
          commitPendingAiMessage();
        }
        if (msg.value === 'idle' && isAudioPlayingRef.current) {
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
            currentAiMessageRef.current += msg.text;
            setMessages((prev) => {
              const copy = [...prev];
              const last = copy[copy.length - 1];
              if (last && last.role === 'ai' && last.partial) {
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

      case 'error':
        clearPlaybackQueue();
        commitPendingAiMessage();
        currentUserMessageRef.current = '';
        setPipelineStatus('idle');
        setMessages((prev) => {
          const next = [...prev];
          const errorText = msg.message || 'Something went wrong in the voice pipeline.';
          const last = next[next.length - 1];
          if (last && last.role === 'ai' && last.text === errorText) {
            return next;
          }
          next.push({
            role: 'ai',
            text: errorText,
            agent: activeAgent,
            partial: false,
          });
          return next;
        });
        break;

      default:
        break;
    }
  }, [activeAgent, clearPlaybackQueue, commitPendingAiMessage, playAudioChunk]);

  const transportHandlers = {
    onMessage: handleMessage,
    onOpen: () => console.log('[Pipeline] Realtime transport connected'),
    onClose: () => console.log('[Pipeline] Realtime transport disconnected'),
  };

  const { status: wsStatus, sendJson, sendBinary } = useWebSocket(
    mode === 'voice' ? '/ws/voice' : '/ws/chat',
    transportHandlers,
    true
  );

  const toggleVoice = useCallback(async () => {
    if (pipelineStatus === 'recording' || pipelineStatus === 'accumulating') {
      stopCapture();
      setPipelineStatus('processing');
      sendJson({ type: 'stop_audio' });
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

  const sendTextMessage = useCallback((text) => {
    if (!text.trim()) return;
    sendJson({ type: 'text', text: text.trim() });
    currentAiMessageRef.current = '';
  }, [sendJson]);

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
    activeAgent,
    pipelineStatus: effectivePipelineStatus,
    messages,
    threadId,
    mode,
    connectionStatus: wsStatus,
    isPlaying,
    isUserSpeaking,
    toggleVoice,
    sendTextMessage,
    switchMode,
  };
}
