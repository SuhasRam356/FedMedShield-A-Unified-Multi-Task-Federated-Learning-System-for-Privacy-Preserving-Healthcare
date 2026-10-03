// ═══════════════════════════════════════════════════════════════
// FedMedShield — Hospital Type Definitions
// Defines hospital node structure, status, and module assignments
// ═══════════════════════════════════════════════════════════════

/** Modules a hospital can participate in */
export type HospitalModule =
  | 'ehr'
  | 'sepsis'
  | 'covid'
  | 'imaging'
  | 'tumor'
  | 'glaucoma'
  | 'drug'
  | 'ids';

/** Operational status of a hospital node */
export type HospitalStatus = 'active' | 'inactive' | 'training' | 'error';

/** A single hospital node in the federated network */
export interface Hospital {
  id: string;
  name: string;
  location: string;
  status: HospitalStatus;
  modules: HospitalModule[];
  lastSeen: Date;
  localAccuracy: number;
  dataSize: number;
}

/** Per-module accuracy breakdown for radar chart */
export interface HospitalModuleMetrics {
  hospitalId: string;
  metrics: Record<HospitalModule, number>;
}

/** Hospital registration / login credentials */
export interface HospitalCredentials {
  username: string;
  password: string;
}

/** JWT auth response after successful login */
export interface AuthResponse {
  accessToken: string;
  tokenType: string;
  hospitalId: string;
  hospitalName: string;
}
