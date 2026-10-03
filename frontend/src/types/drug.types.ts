// ═══════════════════════════════════════════════════════════════
// FedMedShield — Drug Discovery Type Definitions
// Module 3: Drug-protein binding affinity, compound screening
// ═══════════════════════════════════════════════════════════════

/** Drug compound input for screening */
export interface DrugCompoundInput {
  smiles: string;
  targetProtein: string;
  compoundName?: string;
}

/** Drug screening result */
export interface DrugScreenResult {
  compoundId: string;
  bindingAffinity: number;
  viable: boolean;
  targetProtein: string;
  predictedEfficacy: number;
}

/** Available target proteins for dropdown */
export interface TargetProtein {
  id: string;
  name: string;
  uniprotId: string;
  category: string;
}

/** Drug screening history entry */
export interface DrugScreenHistory {
  id: string;
  compoundId: string;
  smiles: string;
  targetProtein: string;
  bindingAffinity: number;
  viable: boolean;
  predictedEfficacy: number;
  timestamp: Date;
  hospitalId: string;
}

/** Unified drug API response wrapper */
export interface DrugResponse {
  success: boolean;
  result: DrugScreenResult;
  modelVersion: string;
  flRound: number;
}
