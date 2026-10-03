// ═══════════════════════════════════════════════════════════════
// FedMedShield — Disease Prediction Type Definitions
// Module 1: EHR multi-task prediction, COVID mortality, sepsis
// ═══════════════════════════════════════════════════════════════

/** Patient input data for disease prediction */
export interface PatientData {
  patientId?: string;
  age: number;
  gender: 'male' | 'female' | 'other';
  bmi: number;
  bloodPressureSystolic: number;
  bloodPressureDiastolic: number;
  glucose: number;
  cholesterol: number;
  hba1c: number;
  creatinine: number;
  wbc: number;
  heartRate: number;
  temperature: number;
  respiratoryRate: number;
  oxygenSaturation: number;
}

/** Multi-disease prediction result */
export interface DiseasePrediction {
  patientId: string;
  hospitalId: string;
  predictions: {
    diabetes: number;
    heartDisease: number;
    cancer: number;
    other: number;
  };
  primaryDiagnosis: string;
  confidence: number;
  timestamp: Date;
}

/** Risk level classification */
export type RiskLevel = 'low' | 'moderate' | 'high' | 'critical';

/** COVID-19 mortality risk assessment */
export interface COVIDMortalityResult {
  mortalityRisk: number;
  riskLevel: RiskLevel;
  keyFactors: string[];
}

/** Sepsis urgency classification */
export type SepsisUrgency = 'normal' | 'watch' | 'warning' | 'critical';

/** Sepsis early warning alert */
export interface SepsisAlert {
  sepsisRisk: number;
  alert: boolean;
  urgency: SepsisUrgency;
  recommendedAction: string;
}

/** Unified prediction API response wrapper */
export interface PredictionResponse {
  success: boolean;
  module: 'disease' | 'covid' | 'sepsis';
  result: DiseasePrediction | COVIDMortalityResult | SepsisAlert;
  modelVersion: string;
  flRound: number;
}
