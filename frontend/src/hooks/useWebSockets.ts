import { useEffect, useRef } from 'react';
import { useStore } from '../store/useStore';

export const useWebSockets = (taskId: number | null) => {
  const ws = useRef<WebSocket | null>(null);
  const updateMetrics = useStore(state => state.updateMetrics);

  useEffect(() => {
    if (!taskId) return;

    // In dev, connect to localhost:8000/ws/{taskId}
    const wsUrl = `ws://127.0.0.1:8000/ws/${taskId}`;
    
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
