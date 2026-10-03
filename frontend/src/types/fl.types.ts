// ═══════════════════════════════════════════════════════════════
// FedMedShield — Federated Learning Type Definitions
// FL status, round history, privacy budget tracking
// ═══════════════════════════════════════════════════════════════

/** Differential privacy budget tracker */
export interface PrivacyBudget {
  epsilon: number;
  delta: number;
  noiseScale: number;
  maxBudget: number;
}

/** Overall FL training status */
export type FLTrainingStatus = 'idle' | 'running' | 'aggregating' | 'completed';

/** Aggregation algorithm choice */
export type FLAlgorithm = 'FedAvg' | 'FedProx' | 'FedNova';

/** Global FL system status — returned by GET /api/fl/status */
export interface FLStatus {
  currentRound: number;
  totalRounds: number;
  globalAccuracy: number;
  globalLoss: number;
  activeHospitals: number;
  totalHospitals: number;
  privacyBudget: PrivacyBudget;
  status: FLTrainingStatus;
}

/** Single FL training round record */
export interface FLRound {
  roundNumber: number;
  timestamp: Date;
  participatingHospitals: string[];
  globalAccuracy: number;
  globalLoss: number;
  aggregationTime: number;
  algorithm: FLAlgorithm;
}

/** FL training configuration sent to POST /api/fl/start */
export interface FLConfig {
  totalRounds: number;
  localEpochs: number;
  minClients: number;
  fractionFit: number;
  algorithm: FLAlgorithm;
  dpEpsilon: number;
  dpDelta: number;
  dpNoiseScale: number;
}

/** Real-time WebSocket FL update payload */
export interface FLWebSocketUpdate {
  event: 'round_start' | 'round_end' | 'aggregation' | 'training_complete' | 'error';
  data: Partial<FLRound> & { message?: string };
}

export interface FLTask {
  id: number;
  name: string;
  task_type: string;
  status: string;
  target_rounds: number;
  current_round?: number;
  num_clients: number;
  dp_epsilon: number;
  created_at?: string;
}

export interface FLTaskCreateRequest {
  name: string;
  task_type: string;
  target_rounds: number;
  num_clients: number;
  dp_epsilon?: number;
}

export interface HospitalNode {
  client_id: string;
  name: string;
  region: string;
  status: 'online' | 'offline' | 'training';
  data_size: string;
  latency_ms: number;
  last_seen: string;
}

export interface FLMetrics {
  current_round: number;
  total_rounds: number;
  progress_percent: number;
  current_loss: number;
  current_accuracy?: number;
}

