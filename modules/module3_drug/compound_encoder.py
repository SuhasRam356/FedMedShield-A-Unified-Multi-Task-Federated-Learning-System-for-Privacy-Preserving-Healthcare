"""
═══════════════════════════════════════════════════════════════
FedMedShield — Compound Encoder (Module 3)
Utility for encoding SMILES strings into molecular fingerprint
vectors. Uses Morgan circular fingerprints when RDKit is available,
falls back to a hash-based encoder otherwise.
═══════════════════════════════════════════════════════════════
"""

import hashlib
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Try to import RDKit (optional dependency)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
    logger.info("RDKit available — using Morgan fingerprints for compound encoding")
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning(
        "RDKit not installed — using hash-based fallback encoder. "
        "For better results, install RDKit: conda install -c conda-forge rdkit"
    )


class CompoundEncoder:
    """
    Encodes drug compounds (SMILES strings) into numerical fingerprint vectors.

    When RDKit is available:
      - Morgan circular fingerprints (radius=2, 2048-bit → compressed to target_dim)
      - Plus molecular descriptors: LogP, MW, TPSA, HBA, HBD, rotatable bonds

    Fallback (no RDKit):
      - SHA-256 hash-based deterministic encoding
      - Character-level n-gram features
      - Statistical properties of the SMILES string

    Args:
        target_dim: Output vector dimensionality (default: 512).
    """

    def __init__(self, target_dim: int = 512) -> None:
        self.target_dim = target_dim
        self.use_rdkit = RDKIT_AVAILABLE

    def encode(self, smiles: str) -> Optional[np.ndarray]:
        """
        Encode a SMILES string into a fixed-length fingerprint vector.

        Args:
            smiles: SMILES molecular representation string.

        Returns:
            Numpy array of shape (target_dim,) or None if encoding fails.
        """
        if self.use_rdkit:
            return self._encode_rdkit(smiles)
        return self._encode_fallback(smiles)

    def _encode_rdkit(self, smiles: str) -> Optional[np.ndarray]:
        """Encode using RDKit Morgan fingerprints + molecular descriptors."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return None

            # Morgan fingerprint (radius=2, 2048 bits)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fp_array = np.array(fp, dtype=np.float32)

            # Compress to first part of target_dim
            fp_dim = self.target_dim - 16  # Reserve 16 dims for descriptors
            if len(fp_array) > fp_dim:
                # Fold the fingerprint to target size
                folded = np.zeros(fp_dim, dtype=np.float32)
                for i, bit in enumerate(fp_array):
                    folded[i % fp_dim] += bit
                fp_compressed = np.clip(folded, 0, 1)
            else:
                fp_compressed = fp_array[:fp_dim]
                if len(fp_compressed) < fp_dim:
                    fp_compressed = np.pad(fp_compressed, (0, fp_dim - len(fp_compressed)))

            # Molecular descriptors (normalized)
            descriptors = np.array([
                Descriptors.MolLogP(mol) / 10.0,
                Descriptors.MolWt(mol) / 1000.0,
                Descriptors.TPSA(mol) / 200.0,
                Descriptors.NumHDonors(mol) / 10.0,
                Descriptors.NumHAcceptors(mol) / 15.0,
                Descriptors.NumRotatableBonds(mol) / 15.0,
                Descriptors.RingCount(mol) / 10.0,
                Descriptors.FractionCSP3(mol),
                Descriptors.HeavyAtomCount(mol) / 50.0,
                Descriptors.NumAromaticRings(mol) / 5.0,
                Descriptors.NumSaturatedRings(mol) / 5.0,
                Descriptors.NumAliphaticRings(mol) / 5.0,
                Descriptors.LabuteASA(mol) / 200.0,
                Descriptors.BalabanJ(mol) / 5.0 if Descriptors.BalabanJ(mol) != 0 else 0.0,
                Descriptors.BertzCT(mol) / 2000.0,
                Descriptors.Ipc(mol) / 10000.0 if Descriptors.Ipc(mol) < 10000 else 1.0,
            ], dtype=np.float32)

            # Clip descriptors to reasonable range
            descriptors = np.clip(descriptors, -1, 1)

            return np.concatenate([fp_compressed, descriptors])

        except Exception as e:
            logger.warning(f"RDKit encoding failed for '{smiles}': {e}")
            return self._encode_fallback(smiles)

    def _encode_fallback(self, smiles: str) -> np.ndarray:
        """
        Hash-based fallback encoder when RDKit is not available.
        Creates a deterministic, reproducible encoding from the SMILES string.
        """
        vector = np.zeros(self.target_dim, dtype=np.float32)

        # 1. SHA-256 hash → first 256 dims
        hash_bytes = hashlib.sha256(smiles.encode()).digest()
        for i, byte in enumerate(hash_bytes):
            if i < self.target_dim:
                vector[i] = byte / 255.0

        # 2. Character frequency features → next dims
        char_offset = 256
        smiles_chars = set("CNOSPFClBrI@=#()[]+-/\\1234567890cnops")
        for j, char in enumerate(sorted(smiles_chars)):
            idx = char_offset + j
            if idx < self.target_dim:
                vector[idx] = smiles.count(char) / max(len(smiles), 1)

        # 3. N-gram features (2-grams and 3-grams)
        ngram_offset = char_offset + len(smiles_chars)
        for n in [2, 3]:
            for i in range(len(smiles) - n + 1):
                ngram = smiles[i:i + n]
                h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
                idx = ngram_offset + (h % (self.target_dim - ngram_offset))
                if idx < self.target_dim:
                    vector[idx] += 1.0 / max(len(smiles), 1)

        # 4. Statistical features in last 16 dims
        stat_offset = self.target_dim - 16
        vector[stat_offset] = len(smiles) / 100.0
        vector[stat_offset + 1] = smiles.count("C") / max(len(smiles), 1)
        vector[stat_offset + 2] = smiles.count("N") / max(len(smiles), 1)
        vector[stat_offset + 3] = smiles.count("O") / max(len(smiles), 1)
        vector[stat_offset + 4] = smiles.count("S") / max(len(smiles), 1)
        vector[stat_offset + 5] = smiles.count("=") / max(len(smiles), 1)
        vector[stat_offset + 6] = smiles.count("(") / max(len(smiles), 1)
        vector[stat_offset + 7] = smiles.count("[") / max(len(smiles), 1)
        vector[stat_offset + 8] = smiles.count("#") / max(len(smiles), 1)
        vector[stat_offset + 9] = smiles.count("@") / max(len(smiles), 1)
        vector[stat_offset + 10] = sum(c.isdigit() for c in smiles) / max(len(smiles), 1)
        vector[stat_offset + 11] = sum(c.isupper() for c in smiles) / max(len(smiles), 1)
        vector[stat_offset + 12] = sum(c.islower() for c in smiles) / max(len(smiles), 1)
        # Ring count approximation (count of digits indicating ring closures)
        vector[stat_offset + 13] = sum(1 for c in smiles if c.isdigit()) / 10.0
        # Branch count
        vector[stat_offset + 14] = smiles.count("(") / 10.0
        # Aromatic atom fraction
        vector[stat_offset + 15] = sum(1 for c in smiles if c.islower()) / max(len(smiles), 1)

        return vector

    def batch_encode(self, smiles_list: list) -> np.ndarray:
        """
        Encode a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Numpy array of shape (N, target_dim).
        """
        vectors = []
        for smiles in smiles_list:
            vec = self.encode(smiles)
            if vec is None:
                vec = np.zeros(self.target_dim, dtype=np.float32)
            vectors.append(vec)
        return np.stack(vectors)
