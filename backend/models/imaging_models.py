"""
Pydantic Schemas for Medical Imaging Classification and Segmentation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ImagingPredictionRequest(BaseModel):
    modality: str = Field(..., example="MRI")  # MRI, CT, Fundus, X-Ray
    target_condition: str = Field(..., example="Brain Tumor")  # Brain Tumor, Glaucoma, Pneumonia
    patient_id: Optional[str] = "IMG-9012"
    image_base64: Optional[str] = None


class DetectedRegion(BaseModel):
    label: str
    confidence: float
    box: List[float]  # [ymin, xmin, ymax, xmax]


class ImagingPredictionResponse(BaseModel):
    patient_id: str
    modality: str
    target_condition: str
    prediction_label: str
    probability: float
    classification: str  # Normal, Malignant, Glaucoma Positive, etc.
    detected_regions: List[DetectedRegion] = []
    heatmap_available: bool = True
    inference_time_ms: float
    disclaimer: str = "RESEARCH DEMONSTRATION ONLY — Not for diagnostic or radiological decision-making. Outputs are simulated demonstration values."
    is_synthetic_simulation: bool = True

