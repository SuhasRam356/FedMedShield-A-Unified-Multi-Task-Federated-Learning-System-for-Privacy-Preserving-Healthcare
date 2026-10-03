// ═══════════════════════════════════════════════════════════════
// FedMedShield — Medical Imaging Type Definitions
// Module 2: Tumor detection, glaucoma, COVID X-ray classification
// ═══════════════════════════════════════════════════════════════

/** Bounding box for detected regions in medical images */
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Tumor classification */
export type TumorType = 'benign' | 'malignant' | 'none';

/** Brain tumor detection result from MRI scan */
export interface TumorDetectionResult {
  detected: boolean;
  type: TumorType;
  confidence: number;
  location?: BoundingBox;
}

/** Glaucoma severity classification */
export type GlaucomaSeverity = 'normal' | 'suspect' | 'mild' | 'moderate' | 'severe';

/** Glaucoma detection result from retinal fundus image */
export interface GlaucomaResult {
  glaucomaDetected: boolean;
  riskScore: number;
  cupDiscRatio: number;
  severity: GlaucomaSeverity;
}

/** COVID X-ray classification labels */
export type CovidXrayClass = 'covid' | 'normal' | 'pneumonia';

/** COVID-19 chest X-ray classification result */
export interface CovidXrayResult {
  classification: CovidXrayClass;
  confidence: number;
  probabilities: Record<CovidXrayClass, number>;
}

/** Unified imaging API response wrapper */
export interface ImagingResponse {
  success: boolean;
  module: 'tumor' | 'glaucoma' | 'covid-xray';
  result: TumorDetectionResult | GlaucomaResult | CovidXrayResult;
  imageId: string;
  processingTimeMs: number;
}

export interface ImagingPredictionResponse {
  patient_id: string;
  modality: string;
  target_condition: string;
  prediction_label: string;
  probability: number;
  classification: string;
  detected_regions: { label: string; confidence: number; box: number[] }[];
  heatmap_available: boolean;
  inference_time_ms: number;
}

