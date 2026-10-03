"""
═══════════════════════════════════════════════════════════════
FedMedShield — Drug Discovery Data Loader (Module 3)
Generates synthetic drug-protein binding affinity data using
Morgan-fingerprint-like molecular descriptors and protein
sequence embeddings. Supports non-IID splitting across hospitals.
═══════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DrugProteinDataset(Dataset):
    """
    PyTorch Dataset for drug-protein binding affinity prediction.
    Each sample consists of a drug molecular fingerprint vector,
    a protein sequence embedding vector, and a continuous binding
    affinity score.
    """

    def __init__(
        self,
        drug_features: np.ndarray,
        protein_features: np.ndarray,
        binding_affinities: np.ndarray,
        viability_labels: np.ndarray,
    ) -> None:
        """
        Args:
            drug_features: Morgan fingerprint vectors (N, 512).
            protein_features: Protein sequence embeddings (N, 256).
            binding_affinities: Continuous affinity scores in [0, 1].
            viability_labels: Binary viability (1 if affinity > threshold).
        """
        self.drug_features = torch.tensor(drug_features, dtype=torch.float32)
        self.protein_features = torch.tensor(protein_features, dtype=torch.float32)
        self.binding_affinities = torch.tensor(binding_affinities, dtype=torch.float32)
        self.viability_labels = torch.tensor(viability_labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.drug_features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "drug_features": self.drug_features[idx],
            "protein_features": self.protein_features[idx],
            "binding_affinity": self.binding_affinities[idx],
            "viability_label": self.viability_labels[idx],
        }


# ═══ Target Protein Database ═══

# Simulated target protein metadata for the UI dropdown
TARGET_PROTEINS = [
    {"id": "P1", "name": "ACE2", "uniprot_id": "Q9BYF1", "category": "Receptor"},
    {"id": "P2", "name": "TMPRSS2", "uniprot_id": "O15393", "category": "Protease"},
    {"id": "P3", "name": "3CLpro (Mpro)", "uniprot_id": "P0DTD1", "category": "Viral Protease"},
    {"id": "P4", "name": "RdRp (nsp12)", "uniprot_id": "P0DTD1", "category": "Polymerase"},
    {"id": "P5", "name": "Spike Protein", "uniprot_id": "P0DTC2", "category": "Viral Surface"},
    {"id": "P6", "name": "EGFR", "uniprot_id": "P00533", "category": "Kinase"},
    {"id": "P7", "name": "HER2", "uniprot_id": "P04626", "category": "Kinase"},
    {"id": "P8", "name": "BRAF", "uniprot_id": "P15056", "category": "Kinase"},
    {"id": "P9", "name": "CDK4/6", "uniprot_id": "P11802", "category": "Cell Cycle"},
    {"id": "P10", "name": "PD-L1", "uniprot_id": "Q9NZQ7", "category": "Immune Checkpoint"},
]


def generate_synthetic_drug_data(
    num_samples: int = 3000,
    drug_dim: int = 512,
    protein_dim: int = 256,
    num_proteins: int = 10,
    viability_threshold: float = 0.6,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic drug-protein binding affinity dataset.

    Drug features simulate Morgan circular fingerprints (sparse binary
    vectors with some continuous descriptor channels). Protein features
    simulate learned sequence embeddings. Binding affinity is computed
    as a learned nonlinear interaction between drug and protein features
    rather than purely random — ensuring the ML model has a learnable signal.

    Args:
        num_samples: Total number of drug-protein pairs.
        drug_dim: Dimensionality of drug fingerprint vectors.
        protein_dim: Dimensionality of protein embedding vectors.
        num_proteins: Number of distinct target proteins.
        viability_threshold: Affinity threshold for viability label.
        random_seed: For reproducibility.

    Returns:
        DataFrame with drug features, protein features, affinity scores, and labels.
    """
    rng = np.random.RandomState(random_seed)

    # Generate drug fingerprint vectors (sparse binary + continuous descriptors)
    # First 256 dims: sparse binary (Morgan-like bits)
    drug_binary = (rng.random((num_samples, drug_dim // 2)) > 0.85).astype(np.float32)
    # Last 256 dims: continuous molecular descriptors (LogP, MW, TPSA, etc.)
    drug_continuous = rng.normal(loc=0, scale=1.0, size=(num_samples, drug_dim // 2)).astype(np.float32)
    drug_features = np.concatenate([drug_binary, drug_continuous], axis=1)

    # Generate protein embeddings — each protein has a prototype vector
    protein_prototypes = rng.normal(loc=0, scale=1.0, size=(num_proteins, protein_dim)).astype(np.float32)
    # Assign each sample to a random target protein
    protein_assignments = rng.randint(0, num_proteins, size=num_samples)
    # Add noise around prototype to simulate sequence variation within a target
    protein_features = np.array([
        protein_prototypes[p_idx] + rng.normal(0, 0.1, size=protein_dim)
        for p_idx in protein_assignments
    ], dtype=np.float32)

    # ═══ Compute Binding Affinity (Learnable Signal) ═══
    # Use a hidden weight matrix to create a nonlinear interaction
    # This ensures the model can actually learn the relationship
    W_drug = rng.normal(0, 0.05, size=(drug_dim, 64)).astype(np.float32)
    W_protein = rng.normal(0, 0.05, size=(protein_dim, 64)).astype(np.float32)

    drug_proj = drug_features @ W_drug        # (N, 64)
    protein_proj = protein_features @ W_protein  # (N, 64)

    # Interaction: element-wise product + sum → scalar score
    interaction = np.sum(drug_proj * protein_proj, axis=1)
    # Add protein-specific bias (some proteins are easier to bind)
    protein_bias = rng.uniform(-1, 1, size=num_proteins)
    interaction += protein_bias[protein_assignments]
    # Add noise
    interaction += rng.normal(0, 0.3, size=num_samples)

    # Sigmoid to get affinity in [0, 1]
    binding_affinities = 1.0 / (1.0 + np.exp(-interaction * 0.1))
    binding_affinities = binding_affinities.clip(0.01, 0.99).astype(np.float32)

    # Viability: compound is viable if affinity exceeds threshold
    viability_labels = (binding_affinities > viability_threshold).astype(np.int32)

    # Predicted efficacy: correlated with affinity but not identical
    predicted_efficacy = (
        binding_affinities * 0.7
        + rng.uniform(0, 0.3, size=num_samples)
    ).clip(0, 1).astype(np.float32)

    # Build DataFrame (store as flat columns for CSV storage)
    data = {}
    for i in range(drug_dim):
        data[f"drug_{i}"] = drug_features[:, i]
    for i in range(protein_dim):
        data[f"protein_{i}"] = protein_features[:, i]
    data["protein_id"] = protein_assignments
    data["binding_affinity"] = binding_affinities
    data["viability"] = viability_labels
    data["predicted_efficacy"] = predicted_efficacy

    df = pd.DataFrame(data)

    logger.info(
        f"Generated {num_samples} synthetic drug-protein pairs. "
        f"Viable rate: {viability_labels.mean():.2%}, "
        f"Mean affinity: {binding_affinities.mean():.3f}"
    )

    return df


def create_non_iid_drug_split(
    df: pd.DataFrame,
    num_hospitals: int = 4,
    alpha: float = 0.5,
    random_seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Split drug data across hospitals non-IID by protein target.
    Different hospitals specialize in different protein targets,
    creating a realistic non-IID scenario where pharma companies
    focus on different therapeutic areas.

    Args:
        df: Full drug-protein DataFrame.
        num_hospitals: Number of hospital nodes.
        alpha: Dirichlet concentration.
        random_seed: Reproducibility seed.

    Returns:
        Dict mapping hospital_id -> local DataFrame.
    """
    rng = np.random.RandomState(random_seed)
    hospital_ids = [f"hospital-{chr(ord('a') + i)}" for i in range(num_hospitals)]
    hospital_indices: Dict[str, List[int]] = {h: [] for h in hospital_ids}

    protein_ids = df["protein_id"].values
    unique_proteins = np.unique(protein_ids)

    # For each protein target, distribute samples across hospitals
    for protein_idx in unique_proteins:
        protein_samples = np.where(protein_ids == protein_idx)[0]
        rng.shuffle(protein_samples)

        proportions = rng.dirichlet(np.repeat(alpha, num_hospitals))
        counts = (proportions * len(protein_samples)).astype(int)
        counts[-1] = len(protein_samples) - counts[:-1].sum()

        start = 0
        for h_idx, count in enumerate(counts):
            hospital_indices[hospital_ids[h_idx]].extend(
                protein_samples[start:start + count].tolist()
            )
            start += count

    result = {}
    for h_id in hospital_ids:
        indices = hospital_indices[h_id]
        rng.shuffle(indices)
        result[h_id] = df.iloc[indices].reset_index(drop=True)
        h_df = result[h_id]
        logger.info(
            f"  {h_id}: {len(h_df)} samples, "
            f"proteins: {sorted(h_df['protein_id'].unique().tolist())}, "
            f"viable rate: {h_df['viability'].mean():.2%}"
        )

    return result


class DrugDataLoaderFactory:
    """
    Factory for creating PyTorch DataLoaders for the drug discovery module.
    Handles synthetic data generation, non-IID splitting by protein target,
    feature normalization, and train/val/test partitioning.
    """

    def __init__(
        self,
        data_dir: str = "data/drug_data",
        drug_dim: int = 512,
        protein_dim: int = 256,
        num_samples: int = 3000,
        num_hospitals: int = 4,
        non_iid_alpha: float = 0.5,
        batch_size: int = 64,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> None:
        self.data_dir = data_dir
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.num_samples = num_samples
        self.num_hospitals = num_hospitals
        self.non_iid_alpha = non_iid_alpha
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.drug_scaler = StandardScaler()
        self.protein_scaler = StandardScaler()
        self._hospital_data: Optional[Dict[str, pd.DataFrame]] = None

    def _drug_cols(self) -> List[str]:
        return [f"drug_{i}" for i in range(self.drug_dim)]

    def _protein_cols(self) -> List[str]:
        return [f"protein_{i}" for i in range(self.protein_dim)]

    def _load_or_generate(self) -> pd.DataFrame:
        """Load existing data or generate synthetic."""
        csv_path = os.path.join(self.data_dir, "drug_data.csv")

        if os.path.exists(csv_path):
            logger.info(f"Loading existing drug data from {csv_path}")
            return pd.read_csv(csv_path)

        logger.info("No existing drug data found. Generating synthetic...")
        df = generate_synthetic_drug_data(
            num_samples=self.num_samples,
            drug_dim=self.drug_dim,
            protein_dim=self.protein_dim,
            random_seed=self.random_seed,
        )
        os.makedirs(self.data_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved synthetic drug data to {csv_path}")
        return df

    def _create_dataset(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = False,
    ) -> DrugProteinDataset:
        """Convert DataFrame partition to DrugProteinDataset."""
        drug_feats = df[self._drug_cols()].values.astype(np.float32)
        protein_feats = df[self._protein_cols()].values.astype(np.float32)

        if fit_scaler:
            drug_feats = self.drug_scaler.fit_transform(drug_feats)
            protein_feats = self.protein_scaler.fit_transform(protein_feats)
        else:
            drug_feats = self.drug_scaler.transform(drug_feats)
            protein_feats = self.protein_scaler.transform(protein_feats)

        return DrugProteinDataset(
            drug_features=drug_feats,
            protein_features=protein_feats,
            binding_affinities=df["binding_affinity"].values.astype(np.float32),
            viability_labels=df["viability"].values.astype(np.float32),
        )

    def get_hospital_dataloaders(
        self, hospital_id: str
    ) -> Dict[str, DataLoader]:
        """
        Get train/val/test DataLoaders for a specific hospital.

        Args:
            hospital_id: e.g. 'hospital-c'

        Returns:
            Dict with 'train', 'val', 'test' DataLoaders.
        """
        if self._hospital_data is None:
            full_df = self._load_or_generate()
            self._hospital_data = create_non_iid_drug_split(
                full_df,
                num_hospitals=self.num_hospitals,
                alpha=self.non_iid_alpha,
                random_seed=self.random_seed,
            )

        if hospital_id not in self._hospital_data:
            raise ValueError(f"Unknown hospital_id '{hospital_id}'.")

        h_df = self._hospital_data[hospital_id]

        # Stratified split by viability label
        train_df, temp_df = train_test_split(
            h_df,
            test_size=self.val_ratio + self.test_ratio,
            random_state=self.random_seed,
            stratify=h_df["viability"],
        )
        relative_test = self.test_ratio / (self.val_ratio + self.test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test,
            random_state=self.random_seed,
            stratify=temp_df["viability"],
        )

        train_ds = self._create_dataset(train_df, fit_scaler=True)
        val_ds = self._create_dataset(val_df)
        test_ds = self._create_dataset(test_df)

        logger.info(
            f"  {hospital_id}/drug — train: {len(train_ds)}, "
            f"val: {len(val_ds)}, test: {len(test_ds)}"
        )

        return {
            "train": DataLoader(train_ds, batch_size=self.batch_size, shuffle=True),
            "val": DataLoader(val_ds, batch_size=self.batch_size, shuffle=False),
            "test": DataLoader(test_ds, batch_size=self.batch_size, shuffle=False),
        }

    def get_centralized_dataloaders(self) -> Dict[str, DataLoader]:
        """Get combined DataLoaders for centralized baseline."""
        full_df = self._load_or_generate()

        train_df, temp_df = train_test_split(
            full_df,
            test_size=self.val_ratio + self.test_ratio,
            random_state=self.random_seed,
            stratify=full_df["viability"],
        )
        relative_test = self.test_ratio / (self.val_ratio + self.test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test,
            random_state=self.random_seed,
            stratify=temp_df["viability"],
        )

        return {
            "train": DataLoader(
                self._create_dataset(train_df, fit_scaler=True),
                batch_size=self.batch_size, shuffle=True,
            ),
            "val": DataLoader(
                self._create_dataset(val_df),
                batch_size=self.batch_size, shuffle=False,
            ),
            "test": DataLoader(
                self._create_dataset(test_df),
                batch_size=self.batch_size, shuffle=False,
            ),
        }
