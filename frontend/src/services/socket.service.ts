type MessageHandler = (data: any) => void;

class SocketService {
  private ws: WebSocket | null = null;
  private listeners: Map<string, Set<MessageHandler>> = new Map();
  private reconnectInterval: number = 3000;
  private currentTaskId: number | null = null;

  public connect(taskId: number) {
    if (this.ws && this.currentTaskId === taskId) {
      return;
    }
    this.currentTaskId = taskId;
    this.disconnect();

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.hostname}:8000/ws/${taskId}`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log(`[SocketService] Connected to FL Task ${taskId}`);
        this.emit('open', { taskId });
      };

      this.ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          this.emit('message', parsed);
          if (parsed.type) {
            this.emit(parsed.type, parsed.payload || parsed);
          }
        } catch (e) {
          console.warn('[SocketService] Non-JSON payload received:', event.data);
        }
      };

      this.ws.onclose = () => {
        console.log('[SocketService] WebSocket closed');
        this.emit('close', {});
      };

      this.ws.onerror = (err) => {
        console.warn('[SocketService] WebSocket error, operating in resilient mode');
        this.emit('error', err);
      };
    } catch (err) {
      console.warn('[SocketService] WebSocket connection failed to establish');
    }
  }

  public subscribe(event: string, handler: MessageHandler) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)?.add(handler);
    return () => this.unsubscribe(event, handler);
  }

  public unsubscribe(event: string, handler: MessageHandler) {
    this.listeners.get(event)?.delete(handler);
  }

  private emit(event: string, data: any) {
    this.listeners.get(event)?.forEach((handler) => handler(data));
  }

  public send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  public disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

export const socketService = new SocketService();
export default socketService;
