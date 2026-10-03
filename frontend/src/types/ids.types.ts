// ═══════════════════════════════════════════════════════════════
// FedMedShield — Intrusion Detection System Type Definitions
// Module 4: Network traffic classification, attack detection
// ═══════════════════════════════════════════════════════════════

/** Network attack type classification (NSL-KDD labels) */
export type AttackType = 'DoS' | 'Probe' | 'R2L' | 'U2R' | 'Normal';

/** Alert severity level */
export type AlertSeverity = 'low' | 'medium' | 'high' | 'critical';

/** Single intrusion detection alert */
export interface IntrusionAlert {
  id: string;
  hospitalId: string;
  timestamp: Date;
  attackType: AttackType;
  severity: AlertSeverity;
  sourceIP: string;
  confidence: number;
  resolved: boolean;
}

/** Network traffic input for scanning */
export interface NetworkTrafficInput {
  duration: number;
  protocolType: 'tcp' | 'udp' | 'icmp';
  service: string;
  flag: string;
  srcBytes: number;
  dstBytes: number;
  land: number;
  wrongFragment: number;
  urgent: number;
  hot: number;
  numFailedLogins: number;
  loggedIn: number;
  numCompromised: number;
  rootShell: number;
  suAttempted: number;
  numRoot: number;
  numFileCreations: number;
  numShells: number;
  numAccessFiles: number;
  count: number;
  srvCount: number;
  serrorRate: number;
  srvSerrorRate: number;
  rerrorRate: number;
  srvRerrorRate: number;
  sameSrvRate: number;
  diffSrvRate: number;
  srvDiffHostRate: number;
  dstHostCount: number;
  dstHostSrvCount: number;
  dstHostSameSrvRate: number;
  dstHostDiffSrvRate: number;
  dstHostSameSrcPortRate: number;
  dstHostSrvDiffHostRate: number;
  dstHostSerrorRate: number;
  dstHostSrvSerrorRate: number;
  dstHostRerrorRate: number;
  dstHostSrvRerrorRate: number;
}

/** IDS dashboard summary stats */
export interface IDSSummary {
  totalAlerts: number;
  resolvedAlerts: number;
  activeThreats: number;
  attackDistribution: Record<AttackType, number>;
  threatsBlockedToday: number;
}

/** Unified IDS API response wrapper */
export interface IDSResponse {
  success: boolean;
  alert: IntrusionAlert;
  summary: IDSSummary;
}
