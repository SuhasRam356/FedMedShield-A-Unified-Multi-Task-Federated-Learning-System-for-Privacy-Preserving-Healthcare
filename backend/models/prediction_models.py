"""
Pydantic Schemas for Clinical Prediction (EHR Multi-Task: Sepsis & COVID-19)
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class PatientDataInput(BaseModel):
    patient_id: Optional[str] = "PT-88219"
    age: int = Field(..., ge=0, le=120, example=58)
    gender: str = Field(..., example="Male")
    heart_rate: float = Field(..., example=98.5)
    systolic_bp: float = Field(..., example=115.0)
    diastolic_bp: float = Field(..., example=72.0)
    respiratory_rate: float = Field(..., example=22.0)
    temperature: float = Field(..., example=38.4)
    spo2: float = Field(..., example=93.5)
    white_blood_cell: float = Field(..., example=14.2)
    platelets: float = Field(..., example=190.0)
    creatinine: float = Field(..., example=1.4)
    crp: float = Field(..., example=45.0)
    comorbidities: List[str] = Field(default=["Hypertension", "Diabetes"])


class PredictionResult(BaseModel):
    patient_id: str
    sepsis_risk_score: float
    sepsis_risk_category: str  # Low, Moderate, High, Critical
    covid_outcome_prob: float
    covid_severity: str  # Mild, Moderate, Severe
    confidence: float
    recommended_interventions: List[str]
    model_version: str
