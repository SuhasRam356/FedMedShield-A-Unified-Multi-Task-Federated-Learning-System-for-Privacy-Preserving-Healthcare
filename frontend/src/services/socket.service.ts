// ═══════════════════════════════════════════════════════════════
// FedMedShield — Real-time Native WebSocket Service
// Uses standard WebSockets conforming to FastAPI's WebSocket protocol
// ═══════════════════════════════════════════════════════════════

type MessageListener = (data: any) => void;

class SocketService {
  private ws: WebSocket | null = null;
  private listeners: Set<MessageListener> = new Set();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 2000;

  connect(channel: string = 'fl-updates'): WebSocket | null {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return this.ws;
    }

    const token = localStorage.getItem('token');
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsBase = import.meta.env.VITE_WS_URL || `${protocol}//${window.location.host}`;
    const wsUrl = `${wsBase}/ws/${channel}${token ? `?token=${encodeURIComponent(token)}` : ''}`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log(`✅ WebSocket connected to channel ${channel}`);
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          this.listeners.forEach((listener) => listener(parsed));
        } catch (e) {
          console.error('Failed to parse WebSocket message', e);
        }
      };

      this.ws.onclose = () => {
        console.log(`⚠️ WebSocket closed for channel ${channel}`);
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts += 1;
          setTimeout(() => this.connect(channel), this.reconnectDelay);
        }
      };

      this.ws.onerror = (error) => {
        console.warn('WebSocket connection error:', error);
      };
    } catch (err) {
      console.warn('Could not establish WebSocket connection:', err);
      return null;
    }

    return this.ws;
  }

  addListener(listener: MessageListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.listeners.clear();
  }
}

export const socketService = new SocketService();
export default socketService;
