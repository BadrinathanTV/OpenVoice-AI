import { useState, useEffect, useRef, useCallback } from 'react';
import { WS_BASE_URL } from '../config/agents';

/**
 * Custom hook for WebSocket connection lifecycle management.
 * Follows SRP: only manages WebSocket connection, not business logic.
 *
 * @param {string} path - WebSocket endpoint path (e.g., '/ws/voice' or '/ws/chat')
 * @param {object} handlers - Message handler callbacks
 * @param {function} handlers.onMessage - Called with parsed JSON messages
 * @param {function} handlers.onOpen - Called when connection opens
 * @param {function} handlers.onClose - Called when connection closes
 * @param {boolean} autoConnect - Whether to connect automatically
 */
export function useWebSocket(path, handlers = {}, autoConnect = true) {
  const [status, setStatus] = useState('disconnected'); // 'connecting' | 'connected' | 'disconnected'
  const wsRef = useRef(null);
  const handlersRef = useRef(handlers);
  const reconnectTimerRef = useRef(null);

  // Keep handlers ref current without re-triggering effects
  useEffect(() => {
    handlersRef.current = handlers;
  }, [handlers]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus('connecting');
    const ws = new WebSocket(`${WS_BASE_URL}${path}`);

    ws.onopen = () => {
      setStatus('connected');
      handlersRef.current.onOpen?.();
    };

    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        try {
          const msg = JSON.parse(event.data);
          handlersRef.current.onMessage?.(msg);
        } catch (e) {
          console.error('[WS] Failed to parse message:', e);
        }
      }
    };

    ws.onclose = () => {
      setStatus('disconnected');
      handlersRef.current.onClose?.();
      wsRef.current = null;

      // Auto-reconnect after 3 seconds
      reconnectTimerRef.current = setTimeout(() => {
        if (autoConnect) connect();
      }, 3000);
    };

    ws.onerror = (err) => {
      console.error('[WS] Error:', err);
    };

    wsRef.current = ws;
  }, [path, autoConnect]);

  const disconnect = useCallback(() => {
    clearTimeout(reconnectTimerRef.current);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus('disconnected');
  }, []);

  /** Send a JSON message. */
  const sendJson = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  /** Send raw binary data (e.g., audio chunks). */
  const sendBinary = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  useEffect(() => {
    if (autoConnect) connect();
    return () => disconnect();
  }, [autoConnect, connect, disconnect]);

  return { status, connect, disconnect, sendJson, sendBinary };
}
