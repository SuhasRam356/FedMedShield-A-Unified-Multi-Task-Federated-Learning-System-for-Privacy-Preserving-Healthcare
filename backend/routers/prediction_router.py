"""
Clinical Disease Prediction Router (EHR Sepsis & COVID-19)
FedMedShield Framework - Privacy-Preserved Multi-Task Inference
"""

import math
from fastapi import APIRouter, HTTPException, status, Depends
from backend.models.prediction_models import PatientDataInput, PredictionResult
from backend.middleware.auth_middleware import get_current_user

router = APIRouter(prefix="/prediction", tags=["Clinical Prediction"])


@router.post("/ehr", response_model=PredictionResult)
async def predict_ehr_risk(
    patient: PatientDataInput,
    current_user: dict = Depends(get_current_user)
):
    """
    Evaluates multi-task risk heuristic in simulated demonstration mode.
    DISCLAIMER: This endpoint runs a synthetic educational surrogate for prototyping.
    It does NOT run validated diagnostic model inference and must not be used for patient care.
    """
    # Calculate physiological deviation score
    temp_dev = max(0.0, patient.temperature - 37.0)
    hr_factor = max(0.0, (patient.heart_rate - 90) / 30.0)
    bp_factor = max(0.0, (90 - patient.systolic_bp) / 40.0)
    wbc_factor = max(0.0, (patient.white_blood_cell - 11.0) / 10.0)
    crp_factor = min(1.0, patient.crp / 100.0)
    spo2_penalty = max(0.0, (95.0 - patient.spo2) / 15.0)

    # Sigmoid sepsis risk calculation (heuristic surrogate)
    raw_sepsis_score = 0.25 * hr_factor + 0.35 * bp_factor + 0.2 * wbc_factor + 0.2 * temp_dev + 0.25 * crp_factor
    sepsis_risk = 1.0 / (1.0 + math.exp(-2.5 * (raw_sepsis_score - 0.4)))
    sepsis_risk = round(min(0.98, max(0.02, sepsis_risk)), 4)

    if sepsis_risk < 0.25:
        sepsis_cat = "Low"
    elif sepsis_risk < 0.60:
        sepsis_cat = "Moderate"
    elif sepsis_risk < 0.85:
        sepsis_cat = "High"
    else:
        sepsis_cat = "Critical"

    # COVID outcome probability (heuristic surrogate)
    raw_covid_score = 0.5 * spo2_penalty + 0.2 * temp_dev + (0.15 if "Diabetes" in patient.comorbidities else 0.0)
    covid_prob = 1.0 / (1.0 + math.exp(-3.0 * (raw_covid_score - 0.3)))
    covid_prob = round(min(0.99, max(0.01, covid_prob)), 4)

    if covid_prob < 0.35:
        covid_sev = "Mild"
    elif covid_prob < 0.70:
        covid_sev = "Moderate"
    else:
        covid_sev = "Severe"

    # Simulation-only educational flags (neutral observations; no prescriptive medical orders)
    interventions = []
    if sepsis_risk > 0.60:
        interventions.append("Simulation Flag: Elevated systemic inflammatory parameters noted in synthetic record.")
        interventions.append("Demonstration Note: Institutional sepsis protocol evaluation indicated in clinical practice.")
    if covid_prob > 0.60 or patient.spo2 < 93.0:
        interventions.append("Simulation Flag: Respiratory biomarker deviation flagged in synthetic scenario.")
    if not interventions:
        interventions.append("Simulation Note: Vital parameters within standard simulated baseline range.")

    return {
        "patient_id": patient.patient_id or "PT-88219",
        "sepsis_risk_score": sepsis_risk,
        "sepsis_risk_category": sepsis_cat,
        "covid_outcome_prob": covid_prob,
        "covid_severity": covid_sev,
        "confidence": 0.942,
        "recommended_interventions": interventions,
        "model_version": "FedMedShield-EHR-Demo-v2.1",
        "disclaimer": "RESEARCH DEMONSTRATION ONLY — Not for clinical or diagnostic decision-making. Outputs are synthetic heuristic estimates and do not constitute medical advice or validated inference.",
        "is_synthetic_simulation": True
    }

