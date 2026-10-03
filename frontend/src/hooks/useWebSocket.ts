import { useEffect, useRef, useState, useCallback } from 'react';

export const useWebSocket = (url: string | null) => {
  const [data, setData] = useState<any>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!url) return;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => setIsConnected(true);
      ws.onclose = () => setIsConnected(false);
      ws.onerror = () => setIsConnected(false);
      ws.onmessage = (event) => {
        try {
          setData(JSON.parse(event.data));
        } catch {
          setData(event.data);
        }
      };

      return () => {
        ws.close();
      };
    } catch {
      setIsConnected(false);
    }
  }, [url]);

  const send = useCallback((message: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(typeof message === 'string' ? message : JSON.stringify(message));
    }
  }, []);

  return { data, isConnected, send };
};

export default useWebSocket;
