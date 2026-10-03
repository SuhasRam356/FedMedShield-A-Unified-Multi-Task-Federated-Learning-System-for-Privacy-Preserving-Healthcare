import apiService from './api.service';

export interface PacketInspectRequest {
  source_ip: string;
  dest_ip: string;
  protocol: string;
  packet_length: number;
  duration_sec: number;
  flag: string;
  bytes_in: number;
  bytes_out: number;
}

export interface IntrusionAlertResponse {
  alert_id: string;
  timestamp: string;
  attack_detected: boolean;
  attack_type: string;
  severity: string;
  confidence: number;
  affected_node: string;
  mitigation_action: string;
}

export const idsService = {
  inspectPacketFlow: async (packet: PacketInspectRequest): Promise<IntrusionAlertResponse> => {
    return apiService.post<IntrusionAlertResponse>('/ids/inspect', packet);
  }
};

export default idsService;
