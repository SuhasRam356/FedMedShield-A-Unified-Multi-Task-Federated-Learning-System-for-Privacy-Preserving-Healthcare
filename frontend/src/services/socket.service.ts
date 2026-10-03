// ═══════════════════════════════════════════════════════════════
// FedMedShield — Real-time Socket & WebSocket Service
// Supports automatic reconnects and polling fallback
// ═══════════════════════════════════════════════════════════════

import { io, Socket } from 'socket.io-client';

class SocketService {
  private socket: Socket | null = null;

  connect(): Socket {
    if (!this.socket) {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      this.socket = io(apiUrl, {
        transports: ['websocket', 'polling'],  // ⬅️ fallback to polling
        reconnection: true,
        reconnectionAttempts: 10,
        reconnectionDelay: 1000,
        timeout: 10000,
      });

      this.socket.on('connect', () => {
        console.log('✅ WebSocket connected:', this.socket?.id);
      });

      this.socket.on('disconnect', (reason) => {
        console.log('⚠️ WebSocket disconnected:', reason);
        // Auto-reconnect if server dropped connection
        if (reason === 'io server disconnect') {
          this.socket?.connect();
        }
      });

      this.socket.on('connect_error', (error) => {
        console.error('❌ WebSocket error:', error.message);
      });
    }
    return this.socket;
  }

  disconnect(): void {
    this.socket?.disconnect();
    this.socket = null;
  }
}

export const socketService = new SocketService();
export default socketService;
