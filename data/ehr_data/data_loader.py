"""
═══════════════════════════════════════════════════════════════
FedMedShield — EHR Data Loader (Module 1)
Generates synthetic Electronic Health Record data with realistic
clinical distributions. Supports non-IID splitting across hospitals.
═══════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EHRDataset(Dataset):
    """
    PyTorch Dataset for Electronic Health Records.
    Supports multi-task labels: disease classification, COVID mortality, sepsis.
    """

    def __init__(
        self,
        features: np.ndarray,
        disease_labels: np.ndarray,
        covid_labels: np.ndarray,
        sepsis_labels: np.ndarray,
    ) -> None:
        """
        Args:
            features: Normalized patient feature matrix (N, num_features).
            disease_labels: Disease class labels (0=healthy, 1=diabetes, 2=heart, 3=cancer).
            covid_labels: Binary COVID mortality labels (0=survived, 1=deceased).
            sepsis_labels: Binary sepsis labels (0=no sepsis, 1=sepsis).
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.disease_labels = torch.tensor(disease_labels, dtype=torch.long)
        self.covid_labels = torch.tensor(covid_labels, dtype=torch.float32)
        self.sepsis_labels = torch.tensor(sepsis_labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "disease_label": self.disease_labels[idx],
            "covid_label": self.covid_labels[idx],
            "sepsis_label": self.sepsis_labels[idx],
        }


def generate_synthetic_ehr_data(
    num_samples: int = 4000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic EHR data with clinically realistic distributions.
    Each feature is drawn from distributions that mirror real-world clinical ranges.

    Args:
        num_samples: Total number of patient records to generate.
        random_seed: Random seed for reproducibility.

    Returns:
        DataFrame with patient features and multi-task labels.
    """
    rng = np.random.RandomState(random_seed)

    # ── Age distribution: skewed toward older patients (hospital population) ──
    age = rng.normal(loc=55, scale=18, size=num_samples).clip(18, 95).astype(int)

    # ── Gender: 0=female, 1=male ──
    gender = rng.binomial(1, 0.52, size=num_samples)

    # ── BMI: Normal distribution centered at 27 (slightly overweight population) ──
    bmi = rng.normal(loc=27.5, scale=6.0, size=num_samples).clip(15.0, 55.0)

    # ── Blood Pressure: systolic and diastolic correlated ──
    bp_systolic = rng.normal(loc=130, scale=22, size=num_samples).clip(80, 220)
    bp_diastolic = (bp_systolic * 0.6 + rng.normal(0, 8, size=num_samples)).clip(50, 140)

    # ── Glucose: fasting blood glucose (mg/dL) ──
    # Bimodal: normal (~90) and diabetic (~160)
    is_high_glucose = rng.binomial(1, 0.3, size=num_samples)
    glucose = np.where(
        is_high_glucose,
        rng.normal(loc=165, scale=40, size=num_samples),
        rng.normal(loc=92, scale=12, size=num_samples),
    ).clip(50, 400)

    # ── Total Cholesterol (mg/dL) ──
    cholesterol = rng.normal(loc=210, scale=42, size=num_samples).clip(100, 400)

    # ── HbA1c (%) — correlated with glucose ──
    hba1c = (glucose * 0.035 + rng.normal(0, 0.5, size=num_samples)).clip(3.5, 14.0)

    # ── Creatinine (mg/dL) — kidney function ──
    creatinine = rng.lognormal(mean=0.1, sigma=0.4, size=num_samples).clip(0.3, 8.0)

    # ── White Blood Cell count (×10³/µL) ──
    wbc = rng.normal(loc=7.5, scale=2.8, size=num_samples).clip(2.0, 30.0)

    # ── Heart Rate (bpm) ──
    heart_rate = rng.normal(loc=78, scale=14, size=num_samples).clip(40, 160)

    # ── Temperature (°F) ──
    temperature = rng.normal(loc=98.6, scale=0.8, size=num_samples).clip(95, 105)

    # ── Respiratory Rate (breaths/min) ──
    respiratory_rate = rng.normal(loc=16, scale=4, size=num_samples).clip(8, 40)

    # ── Oxygen Saturation (%) ──
    oxygen_saturation = rng.normal(loc=96.5, scale=2.5, size=num_samples).clip(70, 100)

    # ── Hemoglobin (g/dL) ──
    hemoglobin = np.where(
        gender == 1,
        rng.normal(loc=15.0, scale=1.8, size=num_samples),
        rng.normal(loc=13.5, scale=1.5, size=num_samples),
    ).clip(6.0, 20.0)

    # ── Platelet Count (×10³/µL) ──
    platelets = rng.normal(loc=250, scale=65, size=num_samples).clip(50, 600)

    # ── Albumin (g/dL) ──
    albumin = rng.normal(loc=3.8, scale=0.6, size=num_samples).clip(1.5, 5.5)

    # ── BUN — Blood Urea Nitrogen (mg/dL) ──
    bun = rng.normal(loc=18, scale=8, size=num_samples).clip(5, 80)

    # ── Smoking status: 0=never, 1=former, 2=current ──
    smoking = rng.choice([0, 1, 2], size=num_samples, p=[0.55, 0.25, 0.20])

    # ── Comorbidity count: 0-5 ──
    comorbidities = rng.poisson(lam=1.5, size=num_samples).clip(0, 8)

    # ═══ Generate Multi-Task Labels Based on Feature Interactions ═══

    # ── Disease Label: logistic model combining risk factors ──
    # 0=healthy, 1=diabetes, 2=heart disease, 3=cancer
    diabetes_risk = (
        0.03 * (glucose - 100)
        + 0.05 * (hba1c - 6.0)
        + 0.02 * (bmi - 25)
        + 0.01 * (age - 40)
        + 0.3 * is_high_glucose
        + rng.normal(0, 0.5, size=num_samples)
    )
    heart_risk = (
        0.02 * (bp_systolic - 120)
        + 0.015 * (cholesterol - 200)
        + 0.01 * (age - 45)
        + 0.03 * (heart_rate - 70)
        + 0.2 * smoking
        + 0.15 * comorbidities
        + rng.normal(0, 0.5, size=num_samples)
    )
    cancer_risk = (
        0.02 * (age - 50)
        + 0.15 * (smoking == 2).astype(float)
        + 0.08 * comorbidities
        - 0.02 * albumin
        + rng.normal(0, 0.8, size=num_samples)
    )

    # Assign disease class based on highest risk exceeding threshold
    disease_labels = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        risks = [0, diabetes_risk[i], heart_risk[i], cancer_risk[i]]
        max_risk_idx = np.argmax(risks)
        if risks[max_risk_idx] > 1.0:
            disease_labels[i] = max_risk_idx
        else:
            disease_labels[i] = 0  # healthy

    # ── COVID Mortality: higher risk for elderly, low O2, high comorbidities ──
    covid_logit = (
        0.04 * (age - 60)
        + 0.3 * comorbidities
        - 0.15 * (oxygen_saturation - 95)
        + 0.1 * (respiratory_rate - 16)
        + 0.05 * (creatinine - 1.0)
        - 0.1 * (albumin - 3.5)
        + rng.normal(0, 1.0, size=num_samples)
    )
    covid_prob = 1.0 / (1.0 + np.exp(-covid_logit))
    covid_labels = (covid_prob > 0.5).astype(int)

    # ── Sepsis: elevated WBC, high temp, high heart rate, low BP ──
    sepsis_logit = (
        0.2 * (wbc - 10)
        + 0.5 * (temperature - 100)
        + 0.05 * (heart_rate - 90)
        - 0.03 * (bp_systolic - 90)
        + 0.1 * (respiratory_rate - 20)
        + 0.15 * (creatinine - 1.2)
        - 0.2 * (platelets / 100 - 2.0)
        + rng.normal(0, 1.0, size=num_samples)
    )
    sepsis_prob = 1.0 / (1.0 + np.exp(-sepsis_logit))
    sepsis_labels = (sepsis_prob > 0.5).astype(int)

    # ═══ Build DataFrame ═══
    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "bp_systolic": bp_systolic,
            "bp_diastolic": bp_diastolic,
            "glucose": glucose,
            "cholesterol": cholesterol,
            "hba1c": hba1c,
            "creatinine": creatinine,
            "wbc": wbc,
            "heart_rate": heart_rate,
            "temperature": temperature,
            "respiratory_rate": respiratory_rate,
            "oxygen_saturation": oxygen_saturation,
            "hemoglobin": hemoglobin,
            "platelets": platelets,
            "albumin": albumin,
            "bun": bun,
            "smoking": smoking,
            "comorbidities": comorbidities,
            # Labels
            "disease_label": disease_labels,
            "covid_label": covid_labels,
            "sepsis_label": sepsis_labels,
        }
    )

    logger.info(
        f"Generated {num_samples} synthetic EHR records. "
        f"Disease dist: {np.bincount(disease_labels, minlength=4).tolist()}, "
        f"COVID mortality rate: {covid_labels.mean():.2%}, "
        f"Sepsis rate: {sepsis_labels.mean():.2%}"
    )

    return df


def create_non_iid_split(
    df: pd.DataFrame,
    num_hospitals: int = 4,
    alpha: float = 0.5,
    random_seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Split data across hospitals using a Dirichlet distribution to simulate
    non-IID (non-identically distributed) data — each hospital gets a
    different disease class distribution.

    Args:
        df: Full patient DataFrame.
        num_hospitals: Number of hospital nodes.
        alpha: Dirichlet concentration parameter.
                Lower alpha → more non-IID (skewed) distribution.
                Higher alpha → more IID (uniform) distribution.
        random_seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping hospital ID to its local DataFrame partition.
    """
    rng = np.random.RandomState(random_seed)
    hospital_ids = [f"hospital-{chr(ord('a') + i)}" for i in range(num_hospitals)]
    hospital_data: Dict[str, List[int]] = {h: [] for h in hospital_ids}

    labels = df["disease_label"].values
    num_classes = len(np.unique(labels))

    # For each class, distribute its samples across hospitals via Dirichlet
    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0]
        rng.shuffle(class_indices)

        # Dirichlet distribution determines proportion per hospital
        proportions = rng.dirichlet(np.repeat(alpha, num_hospitals))
        # Convert proportions to counts (ensuring all samples are assigned)
        proportions = (proportions * len(class_indices)).astype(int)
        # Assign remainder to last hospital
        proportions[-1] = len(class_indices) - proportions[:-1].sum()

        start = 0
        for h_idx, count in enumerate(proportions):
            hospital_data[hospital_ids[h_idx]].extend(
                class_indices[start : start + count].tolist()
            )
            start += count

    # Build per-hospital DataFrames
    result = {}
    for h_id in hospital_ids:
        indices = hospital_data[h_id]
        rng.shuffle(indices)
        h_df = df.iloc[indices].reset_index(drop=True)
        result[h_id] = h_df
        logger.info(
            f"  {h_id}: {len(h_df)} samples, "
            f"disease dist: {np.bincount(h_df['disease_label'].values, minlength=num_classes).tolist()}"
        )

    return result


class EHRDataLoaderFactory:
    """
    Factory class for creating PyTorch DataLoaders for the EHR module.
    Handles data generation, normalization, non-IID splitting, and
    train/val/test partitioning per hospital.
    """

    FEATURE_COLUMNS = [
        "age", "gender", "bmi", "bp_systolic", "bp_diastolic",
        "glucose", "cholesterol", "hba1c", "creatinine", "wbc",
        "heart_rate", "temperature", "respiratory_rate",
        "oxygen_saturation", "hemoglobin", "platelets", "albumin",
        "bun", "smoking", "comorbidities",
    ]
    LABEL_COLUMNS = ["disease_label", "covid_label", "sepsis_label"]

    def __init__(
        self,
        data_dir: str = "data/ehr_data",
        num_samples: int = 4000,
        num_hospitals: int = 4,
        non_iid_alpha: float = 0.5,
        batch_size: int = 32,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> None:
        """
        Args:
            data_dir: Directory to check for / save CSV data.
            num_samples: Total samples if generating synthetic data.
            num_hospitals: Number of hospital nodes.
            non_iid_alpha: Dirichlet alpha for non-IID split.
            batch_size: Batch size for DataLoaders.
            val_ratio: Fraction of each hospital's data for validation.
            test_ratio: Fraction of each hospital's data for testing.
            random_seed: Random seed for reproducibility.
        """
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.num_hospitals = num_hospitals
        self.non_iid_alpha = non_iid_alpha
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self._hospital_data: Optional[Dict[str, pd.DataFrame]] = None

    def _load_or_generate(self) -> pd.DataFrame:
        """Load existing CSV or generate synthetic data."""
        csv_path = os.path.join(self.data_dir, "ehr_data.csv")

        if os.path.exists(csv_path):
            logger.info(f"Loading existing EHR data from {csv_path}")
            df = pd.read_csv(csv_path)
            # Validate required columns exist
            missing = set(self.FEATURE_COLUMNS + self.LABEL_COLUMNS) - set(df.columns)
            if missing:
                logger.warning(
                    f"CSV missing columns {missing}, regenerating synthetic data."
                )
                df = generate_synthetic_ehr_data(self.num_samples, self.random_seed)
                os.makedirs(self.data_dir, exist_ok=True)
                df.to_csv(csv_path, index=False)
        else:
            logger.info("No existing EHR data found. Generating synthetic data...")
            df = generate_synthetic_ehr_data(self.num_samples, self.random_seed)
            os.makedirs(self.data_dir, exist_ok=True)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved synthetic EHR data to {csv_path}")

        return df

    def _normalize_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Normalize feature columns using StandardScaler.

        Args:
            df: DataFrame with feature columns.
            fit: If True, fit the scaler on this data (use for training set).

        Returns:
            Normalized feature array of shape (N, num_features).
        """
        features = df[self.FEATURE_COLUMNS].values.astype(np.float32)
        if fit:
            return self.scaler.fit_transform(features)
        return self.scaler.transform(features)

    def _create_dataset(
        self, df: pd.DataFrame, fit_scaler: bool = False
    ) -> EHRDataset:
        """Convert a DataFrame partition into an EHRDataset."""
        features = self._normalize_features(df, fit=fit_scaler)
        return EHRDataset(
            features=features,
            disease_labels=df["disease_label"].values,
            covid_labels=df["covid_label"].values.astype(np.float32),
            sepsis_labels=df["sepsis_label"].values.astype(np.float32),
        )

    def get_hospital_dataloaders(
        self, hospital_id: str
    ) -> Dict[str, DataLoader]:
        """
        Get train/val/test DataLoaders for a specific hospital.

        Args:
            hospital_id: e.g. 'hospital-a'

        Returns:
            Dictionary with 'train', 'val', 'test' DataLoader objects.
        """
        if self._hospital_data is None:
            full_df = self._load_or_generate()
            self._hospital_data = create_non_iid_split(
                full_df,
                num_hospitals=self.num_hospitals,
                alpha=self.non_iid_alpha,
                random_seed=self.random_seed,
            )

        if hospital_id not in self._hospital_data:
            raise ValueError(
                f"Unknown hospital_id '{hospital_id}'. "
                f"Available: {list(self._hospital_data.keys())}"
            )

        h_df = self._hospital_data[hospital_id]

        # Split into train / val / test
        train_df, temp_df = train_test_split(
            h_df,
            test_size=self.val_ratio + self.test_ratio,
            random_state=self.random_seed,
            stratify=h_df["disease_label"],
        )
        relative_test = self.test_ratio / (self.val_ratio + self.test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test,
            random_state=self.random_seed,
            stratify=temp_df["disease_label"],
        )

        # Fit scaler on training data only
        train_dataset = self._create_dataset(train_df, fit_scaler=True)
        val_dataset = self._create_dataset(val_df, fit_scaler=False)
        test_dataset = self._create_dataset(test_df, fit_scaler=False)

        logger.info(
            f"  {hospital_id} splits — train: {len(train_dataset)}, "
            f"val: {len(val_dataset)}, test: {len(test_dataset)}"
        )

        return {
            "train": DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
            ),
            "val": DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            ),
            "test": DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            ),
        }

    def get_centralized_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Get train/val/test DataLoaders with ALL data combined (for baseline comparison).

        Returns:
            Dictionary with 'train', 'val', 'test' DataLoader objects.
        """
        full_df = self._load_or_generate()

        train_df, temp_df = train_test_split(
            full_df,
            test_size=self.val_ratio + self.test_ratio,
            random_state=self.random_seed,
            stratify=full_df["disease_label"],
        )
        relative_test = self.test_ratio / (self.val_ratio + self.test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test,
            random_state=self.random_seed,
            stratify=temp_df["disease_label"],
        )

        train_dataset = self._create_dataset(train_df, fit_scaler=True)
        val_dataset = self._create_dataset(val_df)
        test_dataset = self._create_dataset(test_df)

        logger.info(
            f"Centralized splits — train: {len(train_dataset)}, "
            f"val: {len(val_dataset)}, test: {len(test_dataset)}"
        )

        return {
            "train": DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
            "val": DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False),
            "test": DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False),
        }

    @property
    def num_features(self) -> int:
        """Number of input features for the EHR model."""
        return len(self.FEATURE_COLUMNS)

    @property
    def num_disease_classes(self) -> int:
        """Number of disease classification classes."""
        return 4  # healthy, diabetes, heart, cancer
