import { useEffect, useRef } from 'react';
import { useStore } from '../store/useStore';

export const useWebSockets = (taskId: number | null) => {
  const ws = useRef<WebSocket | null>(null);
  const updateMetrics = useStore(state => state.updateMetrics);

  useEffect(() => {
    if (!taskId) return;

    const token = localStorage.getItem('token');
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsBase = import.meta.env.VITE_WS_URL || `${protocol}//${window.location.host}`;
    const wsUrl = `${wsBase}/ws/${taskId}${token ? `?token=${encodeURIComponent(token)}` : ''}`;
    
    ws.current = new WebSocket(wsUrl);

    ws.current.onopen = () => {
      console.log(`Connected to WS for task ${taskId}`);
    };

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        updateMetrics(data);
      } catch (e) {
        console.error("Failed to parse WS message", e);
      }
    };

    ws.current.onclose = () => {
      console.log("WS connection closed");
    };

    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [taskId, updateMetrics]);

  return ws.current;
};
