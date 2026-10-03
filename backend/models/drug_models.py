"""
Pydantic Schemas for Federated Drug Discovery & Bioactivity Screening
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class CompoundInput(BaseModel):
    compound_id: str = Field(..., example="CMP-3921")
    smiles_string: str = Field(..., example="CC(=O)Oc1ccccc1C(=O)O")
    target_protein: str = Field(..., example="EGFR_HUMAN")
    molecular_weight: Optional[float] = 180.16


class AffinityPredictionResponse(BaseModel):
    compound_id: str
    target_protein: str
    predicted_kd_nm: float
    binding_affinity_score: float  # -10 to -4 kcal/mol
    bioactivity_class: str  # Highly Active, Active, Inactive
    druggability_probability: float
    safety_admet_flag: str  # Low Toxicity, Moderate Risk, Flagged
    recommendation: str
