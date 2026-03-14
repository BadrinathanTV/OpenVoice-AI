import { useState, useEffect, useRef, useCallback } from 'react';
import { WS_BASE_URL } from '../config/agents';

/**
 * Custom hook for realtime connection lifecycle management.
 * Follows SRP: only manages the transport channel, not business logic.
 *
 * @param {string} path - Transport endpoint path (e.g., '/ws/voice' or '/ws/chat')
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
  const isMountedRef = useRef(false);
  const connectRef = useRef(null);
  const connectionIdRef = useRef(0);

  // Keep handlers ref current without re-triggering effects
  useEffect(() => {
    handlersRef.current = handlers;
  }, [handlers]);

  const connect = useCallback(() => {
    // Guard: don't connect if unmounted or already open
    if (!isMountedRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN ||
        wsRef.current?.readyState === WebSocket.CONNECTING) return;

    setStatus('connecting');
    const connectionId = ++connectionIdRef.current;
    const ws = new WebSocket(`${WS_BASE_URL}${path}`);
    wsRef.current = ws;

    ws.onopen = () => {
      if (connectionId !== connectionIdRef.current || wsRef.current !== ws) {
        ws.close();
        return;
      }
      if (!isMountedRef.current) {
        ws.close();
        return;
      }
      setStatus('connected');
      handlersRef.current.onOpen?.();
    };

    ws.onmessage = (event) => {
      if (connectionId !== connectionIdRef.current || wsRef.current !== ws) {
        return;
      }
      if (typeof event.data === 'string') {
        try {
          const msg = JSON.parse(event.data);
          handlersRef.current.onMessage?.(msg);
        } catch (e) {
          console.error('[Transport] Failed to parse message:', e);
        }
      }
    };

    ws.onclose = () => {
      if (connectionId !== connectionIdRef.current) {
        return;
      }
      setStatus('disconnected');
      handlersRef.current.onClose?.();
      wsRef.current = null;

      // Auto-reconnect only if still mounted
      if (isMountedRef.current && autoConnect) {
        reconnectTimerRef.current = setTimeout(() => {
          if (isMountedRef.current) {
            connectRef.current?.();
          }
        }, 3000);
      }
    };

    ws.onerror = (err) => {
      if (connectionId !== connectionIdRef.current || wsRef.current !== ws) {
        return;
      }
      console.error('[Transport] Error:', err);
    };
  }, [path, autoConnect]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  const disconnect = useCallback(() => {
    connectionIdRef.current += 1;
    clearTimeout(reconnectTimerRef.current);
    reconnectTimerRef.current = null;
    if (wsRef.current) {
      // Remove event handlers before closing to prevent reconnects
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
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
    isMountedRef.current = true;
    let initialConnectTimer = null;

    // Close any stale connection before connecting fresh
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    clearTimeout(reconnectTimerRef.current);

    if (autoConnect) {
      initialConnectTimer = setTimeout(() => {
        if (isMountedRef.current) {
          connectRef.current?.();
        }
      }, 0);
    }

    return () => {
      clearTimeout(initialConnectTimer);
      isMountedRef.current = false;
      disconnect();
    };
  }, [autoConnect, connect, disconnect, path]); // Only reconnect when the path actually changes

  return { status, connect, disconnect, sendJson, sendBinary };
}
