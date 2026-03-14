import { useState, useEffect, useRef, useCallback } from 'react';
import { API_BASE_URL } from '../config/agents';

const ICE_SERVERS = [{ urls: ['stun:stun.l.google.com:19302'] }];

function waitForIceGatheringComplete(peerConnection) {
  if (peerConnection.iceGatheringState === 'complete') {
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    const onStateChange = () => {
      if (peerConnection.iceGatheringState === 'complete') {
        peerConnection.removeEventListener('icegatheringstatechange', onStateChange);
        resolve();
      }
    };
    peerConnection.addEventListener('icegatheringstatechange', onStateChange);
  });
}

/**
 * Realtime voice transport hook powered by WebRTC data channels.
 * Uses one control channel for JSON events and one uplink channel for binary PCM.
 */
export function useWebRTC(offerPath = '/api/webrtc/offer', handlers = {}, autoConnect = true) {
  const [status, setStatus] = useState('disconnected');
  const handlersRef = useRef(handlers);
  const isMountedRef = useRef(false);
  const isConnectingRef = useRef(false);
  const connectionIdRef = useRef(0);
  const reconnectTimerRef = useRef(null);

  const peerConnectionRef = useRef(null);
  const controlChannelRef = useRef(null);
  const audioUplinkChannelRef = useRef(null);
  const connectRef = useRef(null);

  useEffect(() => {
    handlersRef.current = handlers;
  }, [handlers]);

  const scheduleReconnect = useCallback(() => {
    if (!isMountedRef.current || !autoConnect) {
      return;
    }

    clearTimeout(reconnectTimerRef.current);
    reconnectTimerRef.current = setTimeout(() => {
      if (isMountedRef.current) {
        connectRef.current?.();
      }
    }, 3000);
  }, [autoConnect]);

  const disconnect = useCallback(() => {
    connectionIdRef.current += 1;
    clearTimeout(reconnectTimerRef.current);
    reconnectTimerRef.current = null;

    const control = controlChannelRef.current;
    if (control) {
      control.onopen = null;
      control.onmessage = null;
      control.onclose = null;
      control.onerror = null;
      try {
        control.close();
      } catch {
        // Ignore close errors from stale channels.
      }
      controlChannelRef.current = null;
    }

    const audioUplink = audioUplinkChannelRef.current;
    if (audioUplink) {
      audioUplink.onopen = null;
      audioUplink.onclose = null;
      audioUplink.onerror = null;
      try {
        audioUplink.close();
      } catch {
        // Ignore close errors from stale channels.
      }
      audioUplinkChannelRef.current = null;
    }

    const peerConnection = peerConnectionRef.current;
    if (peerConnection) {
      peerConnection.onconnectionstatechange = null;
      peerConnection.oniceconnectionstatechange = null;
      peerConnection.ondatachannel = null;
      try {
        peerConnection.close();
      } catch {
        // Ignore close errors from stale peer connections.
      }
      peerConnectionRef.current = null;
    }

    setStatus('disconnected');
  }, []);

  const connect = useCallback(async () => {
    if (!isMountedRef.current) {
      return;
    }
    if (isConnectingRef.current) {
      return;
    }

    if (!window.RTCPeerConnection) {
      console.error('[Transport] WebRTC is not supported in this browser.');
      setStatus('disconnected');
      return;
    }

    isConnectingRef.current = true;
    const currentConnectionId = ++connectionIdRef.current;
    clearTimeout(reconnectTimerRef.current);
    reconnectTimerRef.current = null;

    // Tear down any previous transport before creating a new one.
    disconnect();
    connectionIdRef.current = currentConnectionId;
    setStatus('connecting');

    const peerConnection = new RTCPeerConnection({ iceServers: ICE_SERVERS });
    peerConnectionRef.current = peerConnection;

    const controlChannel = peerConnection.createDataChannel('ov-control', { ordered: true });
    const audioUplinkChannel = peerConnection.createDataChannel('ov-audio-up', { ordered: true });
    controlChannelRef.current = controlChannel;
    audioUplinkChannelRef.current = audioUplinkChannel;

    const isCurrent = () =>
      isMountedRef.current &&
      currentConnectionId === connectionIdRef.current &&
      peerConnectionRef.current === peerConnection;

    controlChannel.onopen = () => {
      if (!isCurrent()) {
        return;
      }
      setStatus('connected');
      handlersRef.current.onOpen?.();
    };

    controlChannel.onmessage = (event) => {
      if (!isCurrent()) {
        return;
      }
      if (typeof event.data !== 'string') {
        return;
      }
      try {
        const message = JSON.parse(event.data);
        handlersRef.current.onMessage?.(message);
      } catch (error) {
        console.error('[Transport] Failed to parse control message:', error);
      }
    };

    controlChannel.onerror = (error) => {
      if (!isCurrent()) {
        return;
      }
      console.error('[Transport] Control channel error:', error);
    };

    controlChannel.onclose = () => {
      if (!isCurrent()) {
        return;
      }
      setStatus('disconnected');
      handlersRef.current.onClose?.();
      scheduleReconnect();
    };

    audioUplinkChannel.onerror = (error) => {
      if (!isCurrent()) {
        return;
      }
      console.error('[Transport] Audio uplink channel error:', error);
    };

    peerConnection.onconnectionstatechange = () => {
      if (!isCurrent()) {
        return;
      }
      const state = peerConnection.connectionState;
      if (state === 'connected') {
        setStatus('connected');
        return;
      }
      if (state === 'connecting') {
        setStatus('connecting');
        return;
      }
      // "disconnected" can be transient during ICE checks; don't immediately reconnect.
      if (state === 'disconnected') {
        setStatus('connecting');
        return;
      }
      if (state === 'failed' || state === 'closed') {
        setStatus('disconnected');
        handlersRef.current.onClose?.();
        scheduleReconnect();
      }
    };

    try {
      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);
      await waitForIceGatheringComplete(peerConnection);

      if (!isCurrent()) {
        return;
      }

      const response = await fetch(`${API_BASE_URL}${offerPath}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: peerConnection.localDescription?.sdp,
          type: peerConnection.localDescription?.type,
        }),
      });

      if (!response.ok) {
        throw new Error(`Offer failed with status ${response.status}`);
      }

      const answer = await response.json();
      await peerConnection.setRemoteDescription(
        new RTCSessionDescription({
          type: answer.type,
          sdp: answer.sdp,
        })
      );
    } catch (error) {
      if (isCurrent()) {
        console.error('[Transport] WebRTC connection failed:', error);
        setStatus('disconnected');
        handlersRef.current.onClose?.();
        scheduleReconnect();
      }
    } finally {
      isConnectingRef.current = false;
    }
  }, [disconnect, offerPath, scheduleReconnect]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  const sendJson = useCallback((payload) => {
    const channel = controlChannelRef.current;
    if (channel?.readyState === 'open') {
      channel.send(JSON.stringify(payload));
    }
  }, []);

  const sendBinary = useCallback((payload) => {
    const channel = audioUplinkChannelRef.current;
    if (channel?.readyState === 'open') {
      channel.send(payload);
    }
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    let initialConnectTimer = null;

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
  }, [autoConnect, disconnect]);

  return {
    status,
    connect,
    disconnect,
    sendJson,
    sendBinary,
  };
}
