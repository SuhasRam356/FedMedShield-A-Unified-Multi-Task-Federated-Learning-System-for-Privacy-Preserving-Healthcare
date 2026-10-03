"""
═══════════════════════════════════════════════════════════════
FedMedShield — Network Intrusion Detection Data Loader (Module 4)
Loads the NSL-KDD dataset or generates synthetic network traffic
data. Supports non-IID splitting across hospital nodes to simulate
different network traffic patterns per institution.
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

# ═══ NSL-KDD Feature Definitions ═══

# The 41 standard NSL-KDD features (for reference and column naming)
NSL_KDD_FEATURES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

# Attack type mapping: NSL-KDD sub-classes → 5 categories
ATTACK_MAP = {
    "normal": "Normal",
    "neptune": "DoS", "smurf": "DoS", "pod": "DoS", "teardrop": "DoS",
    "land": "DoS", "back": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "mailbomb": "DoS", "processtable": "DoS",
    "ipsweep": "Probe", "portsweep": "Probe", "nmap": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L",
    "phf": "R2L", "multihop": "R2L", "warezmaster": "R2L",
    "warezclient": "R2L", "spy": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "worm": "R2L",
    "httptunnel": "R2L",
    "buffer_overflow": "U2R", "rootkit": "U2R", "loadmodule": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

ATTACK_CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# Protocol and service encoding for synthetic data
PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = [
    "http", "smtp", "ftp", "ftp_data", "ssh", "telnet",
    "domain_u", "private", "finger", "other", "eco_i",
]
FLAGS = ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "S3", "OTH"]


class NetworkTrafficDataset(Dataset):
    """
    PyTorch Dataset for network traffic classification.
    Each sample has 41 numerical features and a 5-class attack label.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Args:
            features: Normalized network traffic features (N, num_features).
            labels: Attack class labels (0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R).
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "label": self.labels[idx],
        }


def load_nsl_kdd(data_dir: str) -> Optional[pd.DataFrame]:
    """
    Attempt to load the real NSL-KDD dataset from disk.
    Expects KDDTrain+.txt or KDDTrain+.csv in data_dir.

    Args:
        data_dir: Directory to search for NSL-KDD files.

    Returns:
        DataFrame if found and loaded, None otherwise.
    """
    possible_names = [
        "KDDTrain+.txt", "KDDTrain+.csv",
        "kddtrain+.txt", "kddtrain+.csv",
        "nsl_kdd_train.csv", "nsl-kdd.csv",
    ]

    for fname in possible_names:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            try:
                # NSL-KDD has 43 columns: 41 features + attack_type + difficulty_level
                col_names = NSL_KDD_FEATURES + ["attack_type", "difficulty_level"]
                df = pd.read_csv(fpath, header=None, names=col_names)
                logger.info(f"Loaded real NSL-KDD data from {fpath}: {len(df)} samples")
                return df
            except Exception as e:
                logger.warning(f"Failed to parse {fpath}: {e}")

    return None


def preprocess_nsl_kdd(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess NSL-KDD DataFrame: encode categoricals with stable schema, map attack labels.

    Args:
        df: Raw NSL-KDD DataFrame.

    Returns:
        Tuple of (feature matrix of shape [N, 41], label array).
    """
    # Map detailed attack types to 5 categories
    df["label"] = df["attack_type"].str.strip().str.lower().map(ATTACK_MAP)
    # Handle unmapped attacks as "Probe" with explicit warning (do not silently classify as Normal)
    unmapped_count = int(df["label"].isna().sum())
    if unmapped_count > 0:
        logger.warning(
            f"Found {unmapped_count} unmapped attack types in NSL-KDD. "
            f"Classifying as anomalous 'Probe' category, not 'Normal'."
        )
        df["label"] = df["label"].fillna("Probe")
        
    label_encoder = LabelEncoder()
    label_encoder.fit(ATTACK_CLASSES)
    labels = label_encoder.transform(df["label"].values)

    # Encode categorical features with fixed schema matching 41 features
    categorical_cols = ["protocol_type", "service", "flag"]
    cat_vocab = {
        "protocol_type": {p: float(i) for i, p in enumerate(PROTOCOLS)},
        "service": {s: float(i) for i, s in enumerate(SERVICES)},
        "flag": {f: float(i) for i, f in enumerate(FLAGS)},
    }
    df_encoded = df[NSL_KDD_FEATURES].copy()
    for col in categorical_cols:
        df_encoded[col] = (
            df_encoded[col]
            .astype(str)
            .str.lower()
            .map(cat_vocab[col])
            .fillna(float(len(cat_vocab[col])))
            .astype(np.float32)
        )

    features = df_encoded.values.astype(np.float32)
    return features, labels


def generate_synthetic_network_data(
    num_samples: int = 5000,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic network traffic data with realistic attack patterns.
    Each attack type has distinct statistical signatures to ensure
    the model can learn meaningful patterns.

    Args:
        num_samples: Total samples to generate.
        random_seed: For reproducibility.

    Returns:
        Tuple of (feature matrix, label array).
    """
    rng = np.random.RandomState(random_seed)

    # Class distribution: heavily imbalanced (realistic)
    class_probs = [0.45, 0.30, 0.12, 0.08, 0.05]  # Normal, DoS, Probe, R2L, U2R
    class_counts = np.array([int(p * num_samples) for p in class_probs])
    class_counts[-1] = num_samples - class_counts[:-1].sum()

    all_features = []
    all_labels = []

    # Number of continuous features we'll generate
    num_features = 41

    for class_idx, count in enumerate(class_counts):
        class_name = ATTACK_CLASSES[class_idx]

        if class_name == "Normal":
            # Normal traffic: moderate duration, standard bytes, low error rates
            features = np.column_stack([
                rng.exponential(scale=50, size=count),        # duration
                rng.choice([0, 1, 2], size=count, p=[0.7, 0.2, 0.1]),  # protocol_type
                rng.randint(0, len(SERVICES), size=count),    # service
                rng.choice(range(len(FLAGS)), size=count, p=[0.7, 0.1, 0.05, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01]),  # flag
                rng.lognormal(mean=6, sigma=2, size=count),   # src_bytes
                rng.lognormal(mean=7, sigma=2, size=count),   # dst_bytes
                np.zeros(count),                               # land
                rng.binomial(1, 0.01, size=count),            # wrong_fragment
                np.zeros(count),                               # urgent
                rng.poisson(0.1, size=count),                 # hot
                np.zeros(count),                               # num_failed_logins
                rng.binomial(1, 0.8, size=count),             # logged_in
                np.zeros(count),                               # num_compromised
                np.zeros(count),                               # root_shell
                np.zeros(count),                               # su_attempted
                np.zeros(count),                               # num_root
                rng.poisson(0.5, size=count),                 # num_file_creations
                np.zeros(count),                               # num_shells
                rng.poisson(0.3, size=count),                 # num_access_files
                np.zeros(count),                               # num_outbound_cmds
                np.zeros(count),                               # is_host_login
                rng.binomial(1, 0.05, size=count),            # is_guest_login
                rng.poisson(10, size=count),                   # count
                rng.poisson(8, size=count),                    # srv_count
                rng.uniform(0, 0.1, size=count),              # serror_rate
                rng.uniform(0, 0.1, size=count),              # srv_serror_rate
                rng.uniform(0, 0.05, size=count),             # rerror_rate
                rng.uniform(0, 0.05, size=count),             # srv_rerror_rate
                rng.uniform(0.8, 1.0, size=count),            # same_srv_rate
                rng.uniform(0, 0.2, size=count),              # diff_srv_rate
                rng.uniform(0, 0.1, size=count),              # srv_diff_host_rate
                rng.randint(1, 256, size=count),              # dst_host_count
                rng.randint(1, 256, size=count),              # dst_host_srv_count
                rng.uniform(0.7, 1.0, size=count),            # dst_host_same_srv_rate
                rng.uniform(0, 0.3, size=count),              # dst_host_diff_srv_rate
                rng.uniform(0, 0.5, size=count),              # dst_host_same_src_port_rate
                rng.uniform(0, 0.1, size=count),              # dst_host_srv_diff_host_rate
                rng.uniform(0, 0.1, size=count),              # dst_host_serror_rate
                rng.uniform(0, 0.1, size=count),              # dst_host_srv_serror_rate
                rng.uniform(0, 0.05, size=count),             # dst_host_rerror_rate
                rng.uniform(0, 0.05, size=count),             # dst_host_srv_rerror_rate
            ])

        elif class_name == "DoS":
            # DoS: very short duration, high src_bytes, high count, high serror_rate
            features = np.column_stack([
                rng.exponential(scale=2, size=count),          # duration (very short)
                rng.choice([0, 1, 2], size=count, p=[0.8, 0.15, 0.05]),
                rng.choice([0, 7], size=count),                # mostly http or private
                rng.choice(range(len(FLAGS)), size=count, p=[0.2, 0.5, 0.1, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02]),
                rng.lognormal(mean=10, sigma=2, size=count),  # high src_bytes
                rng.lognormal(mean=3, sigma=1.5, size=count), # low dst_bytes
                rng.binomial(1, 0.02, size=count),
                rng.binomial(1, 0.05, size=count),
                np.zeros(count),
                rng.poisson(0.2, size=count),
                np.zeros(count),
                rng.binomial(1, 0.1, size=count),             # rarely logged in
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
                rng.poisson(300, size=count),                  # very high count
                rng.poisson(5, size=count),
                rng.uniform(0.7, 1.0, size=count),            # high serror_rate
                rng.uniform(0.7, 1.0, size=count),
                rng.uniform(0, 0.1, size=count),
                rng.uniform(0, 0.1, size=count),
                rng.uniform(0.1, 0.5, size=count),
                rng.uniform(0.3, 0.9, size=count),
                rng.uniform(0, 0.2, size=count),
                rng.randint(200, 256, size=count),            # high dst_host_count
                rng.randint(1, 50, size=count),
                rng.uniform(0.1, 0.5, size=count),
                rng.uniform(0.3, 0.9, size=count),
                rng.uniform(0, 0.3, size=count),
                rng.uniform(0.2, 0.8, size=count),
                rng.uniform(0.6, 1.0, size=count),            # high serror rates
                rng.uniform(0.6, 1.0, size=count),
                rng.uniform(0, 0.1, size=count),
                rng.uniform(0, 0.1, size=count),
            ])

        elif class_name == "Probe":
            # Probe: moderate duration, very high dst_host_count, scanning patterns
            features = np.column_stack([
                rng.exponential(scale=5, size=count),
                rng.choice([0, 1, 2], size=count, p=[0.5, 0.3, 0.2]),
                rng.randint(0, len(SERVICES), size=count),
                rng.choice(range(len(FLAGS)), size=count, p=[0.3, 0.2, 0.2, 0.1, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03]),
                rng.lognormal(mean=4, sigma=2, size=count),
                rng.lognormal(mean=4, sigma=2, size=count),
                np.zeros(count),
                rng.binomial(1, 0.03, size=count),
                np.zeros(count),
                rng.poisson(0.5, size=count),
                rng.binomial(1, 0.1, size=count),
                rng.binomial(1, 0.3, size=count),
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
                np.zeros(count),
                rng.poisson(50, size=count),                   # moderate count
                rng.poisson(3, size=count),                    # low srv_count (many diff services)
                rng.uniform(0, 0.3, size=count),
                rng.uniform(0, 0.3, size=count),
                rng.uniform(0.3, 0.8, size=count),            # high rerror_rate (rejected)
                rng.uniform(0.3, 0.8, size=count),
                rng.uniform(0.05, 0.3, size=count),           # low same_srv_rate (scanning)
                rng.uniform(0.5, 1.0, size=count),            # high diff_srv_rate
                rng.uniform(0.3, 0.9, size=count),            # high srv_diff_host_rate
                rng.randint(150, 256, size=count),             # high dst_host_count
                rng.randint(1, 30, size=count),
                rng.uniform(0.05, 0.3, size=count),
                rng.uniform(0.5, 1.0, size=count),
                rng.uniform(0, 0.2, size=count),
                rng.uniform(0.3, 0.9, size=count),
                rng.uniform(0, 0.3, size=count),
                rng.uniform(0, 0.3, size=count),
                rng.uniform(0.3, 0.8, size=count),
                rng.uniform(0.3, 0.8, size=count),
            ])

        elif class_name == "R2L":
            # R2L: longer duration, failed logins, guest access
            features = np.column_stack([
                rng.lognormal(mean=4, sigma=2, size=count),   # longer duration
                rng.choice([0, 1, 2], size=count, p=[0.8, 0.15, 0.05]),
                rng.choice([0, 2, 5], size=count),            # http, ftp, telnet
                rng.choice(range(len(FLAGS)), size=count, p=[0.5, 0.15, 0.15, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.04]),
                rng.lognormal(mean=8, sigma=2, size=count),
                rng.lognormal(mean=6, sigma=2, size=count),
                np.zeros(count),
                rng.binomial(1, 0.02, size=count),
                np.zeros(count),
                rng.poisson(2, size=count),                    # elevated hot
                rng.poisson(3, size=count),                    # failed logins!
                rng.binomial(1, 0.4, size=count),
                rng.poisson(1, size=count),                    # some compromised
                rng.binomial(1, 0.1, size=count),
                rng.binomial(1, 0.1, size=count),
                rng.poisson(0.5, size=count),
                rng.poisson(0.5, size=count),
                np.zeros(count),
                rng.poisson(1, size=count),
                np.zeros(count),
                rng.binomial(1, 0.05, size=count),
                rng.binomial(1, 0.3, size=count),             # guest login attempts
                rng.poisson(5, size=count),
                rng.poisson(5, size=count),
                rng.uniform(0, 0.2, size=count),
                rng.uniform(0, 0.2, size=count),
                rng.uniform(0.1, 0.5, size=count),
                rng.uniform(0.1, 0.5, size=count),
                rng.uniform(0.5, 0.9, size=count),
                rng.uniform(0.1, 0.5, size=count),
                rng.uniform(0, 0.3, size=count),
                rng.randint(1, 100, size=count),
                rng.randint(1, 100, size=count),
                rng.uniform(0.3, 0.8, size=count),
                rng.uniform(0.1, 0.5, size=count),
                rng.uniform(0.1, 0.6, size=count),
                rng.uniform(0, 0.3, size=count),
                rng.uniform(0, 0.2, size=count),
                rng.uniform(0, 0.2, size=count),
                rng.uniform(0.1, 0.4, size=count),
                rng.uniform(0.1, 0.4, size=count),
            ])

        else:  # U2R
            # U2R: normal-looking traffic but with root_shell, su_attempted indicators
            features = np.column_stack([
                rng.lognormal(mean=3, sigma=1.5, size=count),
                rng.choice([0, 1, 2], size=count, p=[0.7, 0.2, 0.1]),
                rng.choice([0, 5, 7], size=count),
                rng.choice(range(len(FLAGS)), size=count, p=[0.6, 0.1, 0.1, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.04]),
                rng.lognormal(mean=7, sigma=2, size=count),
                rng.lognormal(mean=7, sigma=2, size=count),
                np.zeros(count),
                rng.binomial(1, 0.02, size=count),
                rng.binomial(1, 0.1, size=count),             # urgent!
                rng.poisson(3, size=count),                    # high hot
                rng.poisson(1, size=count),
                rng.binomial(1, 0.7, size=count),
                rng.poisson(2, size=count),                    # compromised
                rng.binomial(1, 0.5, size=count),             # root_shell!
                rng.binomial(1, 0.4, size=count),             # su_attempted!
                rng.poisson(3, size=count),                    # num_root!
                rng.poisson(2, size=count),
                rng.poisson(1, size=count),                    # shells
                rng.poisson(2, size=count),
                np.zeros(count),
                np.zeros(count),
                rng.binomial(1, 0.1, size=count),
                rng.poisson(3, size=count),
                rng.poisson(3, size=count),
                rng.uniform(0, 0.15, size=count),
                rng.uniform(0, 0.15, size=count),
                rng.uniform(0, 0.1, size=count),
                rng.uniform(0, 0.1, size=count),
                rng.uniform(0.6, 1.0, size=count),
                rng.uniform(0, 0.3, size=count),
                rng.uniform(0, 0.2, size=count),
                rng.randint(1, 200, size=count),
                rng.randint(1, 200, size=count),
                rng.uniform(0.5, 1.0, size=count),
                rng.uniform(0, 0.3, size=count),
                rng.uniform(0, 0.4, size=count),
                rng.uniform(0, 0.2, size=count),
                rng.uniform(0, 0.15, size=count),
                rng.uniform(0, 0.15, size=count),
                rng.uniform(0, 0.1, size=count),
                rng.uniform(0, 0.1, size=count),
            ])

        all_features.append(features)
        all_labels.append(np.full(count, class_idx, dtype=np.int64))

    features = np.vstack(all_features).astype(np.float32)
    labels = np.concatenate(all_labels)

    # Shuffle all samples together
    shuffle_idx = rng.permutation(len(features))
    features = features[shuffle_idx]
    labels = labels[shuffle_idx]

    logger.info(
        f"Generated {len(features)} synthetic network traffic samples. "
        f"Class dist: {dict(zip(ATTACK_CLASSES, np.bincount(labels, minlength=5).tolist()))}"
    )

    return features, labels


def create_non_iid_network_split(
    features: np.ndarray,
    labels: np.ndarray,
    num_hospitals: int = 4,
    alpha: float = 0.5,
    random_seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split network data across hospitals non-IID using Dirichlet distribution.
    Different hospitals see different attack type distributions.

    Args:
        features: Feature matrix (N, D).
        labels: Label array.
        num_hospitals: Number of hospital nodes.
        alpha: Dirichlet concentration.
        random_seed: Reproducibility.

    Returns:
        Dict mapping hospital_id -> (features, labels) tuple.
    """
    rng = np.random.RandomState(random_seed)
    hospital_ids = [f"hospital-{chr(ord('a') + i)}" for i in range(num_hospitals)]
    hospital_indices: Dict[str, List[int]] = {h: [] for h in hospital_ids}

    num_classes = len(ATTACK_CLASSES)

    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0]
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(np.repeat(alpha, num_hospitals))
        counts = (proportions * len(class_indices)).astype(int)
        counts[-1] = len(class_indices) - counts[:-1].sum()

        start = 0
        for h_idx, count in enumerate(counts):
            hospital_indices[hospital_ids[h_idx]].extend(
                class_indices[start:start + count].tolist()
            )
            start += count

    result = {}
    for h_id in hospital_ids:
        indices = np.array(hospital_indices[h_id])
        rng.shuffle(indices)
        result[h_id] = (features[indices], labels[indices])
        logger.info(
            f"  {h_id}: {len(indices)} samples, "
            f"class dist: {dict(zip(ATTACK_CLASSES, np.bincount(labels[indices], minlength=num_classes).tolist()))}"
        )

    return result


class NetworkDataLoaderFactory:
    """
    Factory for creating PyTorch DataLoaders for the IDS module.
    Handles NSL-KDD loading or synthetic generation, non-IID splitting,
    feature normalization, and train/val/test partitioning.
    """

    def __init__(
        self,
        data_dir: str = "data/network_data",
        num_samples: int = 5000,
        num_hospitals: int = 4,
        non_iid_alpha: float = 0.5,
        batch_size: int = 64,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> None:
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.num_hospitals = num_hospitals
        self.non_iid_alpha = non_iid_alpha
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self._hospital_data: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None

    @property
    def num_features(self) -> int:
        """Number of input features."""
        return 41

    @property
    def num_classes(self) -> int:
        """Number of attack classes."""
        return len(ATTACK_CLASSES)

    def _load_or_generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load real NSL-KDD or generate synthetic data."""
        # Try real data
        real_df = load_nsl_kdd(self.data_dir)
        if real_df is not None:
            return preprocess_nsl_kdd(real_df)

        # Generate synthetic
        logger.info("No NSL-KDD data found. Generating synthetic network data...")
        return generate_synthetic_network_data(
            num_samples=self.num_samples,
            random_seed=self.random_seed,
        )

    def get_hospital_dataloaders(
        self, hospital_id: str
    ) -> Dict[str, DataLoader]:
        """
        Get train/val/test DataLoaders for a specific hospital.

        Args:
            hospital_id: e.g. 'hospital-d'

        Returns:
            Dict with 'train', 'val', 'test' DataLoaders.
        """
        if self._hospital_data is None:
            features, labels = self._load_or_generate()
            self._hospital_data = create_non_iid_network_split(
                features, labels,
                num_hospitals=self.num_hospitals,
                alpha=self.non_iid_alpha,
                random_seed=self.random_seed,
            )

        if hospital_id not in self._hospital_data:
            raise ValueError(f"Unknown hospital_id '{hospital_id}'.")

        h_features, h_labels = self._hospital_data[hospital_id]

        # Train/val/test split
        indices = np.arange(len(h_features))
        unique_classes, counts = np.unique(h_labels, return_counts=True)
        can_stratify = len(unique_classes) > 1 and int(np.min(counts)) >= 2

        train_idx, temp_idx = train_test_split(
            indices,
            test_size=self.val_ratio + self.test_ratio,
            random_state=self.random_seed,
            stratify=h_labels if can_stratify else None,
        )
        relative_test = self.test_ratio / (self.val_ratio + self.test_ratio)
        temp_classes, temp_counts = np.unique(h_labels[temp_idx], return_counts=True)
        can_stratify_temp = len(temp_classes) > 1 and int(np.min(temp_counts)) >= 2

        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test,
            random_state=self.random_seed,
            stratify=h_labels[temp_idx] if can_stratify_temp else None,
        )

        # Fit scaler on training data only
        train_features = self.scaler.fit_transform(h_features[train_idx])
        val_features = self.scaler.transform(h_features[val_idx])
        test_features = self.scaler.transform(h_features[test_idx])

        train_ds = NetworkTrafficDataset(train_features, h_labels[train_idx])
        val_ds = NetworkTrafficDataset(val_features, h_labels[val_idx])
        test_ds = NetworkTrafficDataset(test_features, h_labels[test_idx])

        logger.info(
            f"  {hospital_id}/ids — train: {len(train_ds)}, "
            f"val: {len(val_ds)}, test: {len(test_ds)}"
        )

        return {
            "train": DataLoader(train_ds, batch_size=self.batch_size, shuffle=True),
            "val": DataLoader(val_ds, batch_size=self.batch_size, shuffle=False),
            "test": DataLoader(test_ds, batch_size=self.batch_size, shuffle=False),
        }

    def get_centralized_dataloaders(self) -> Dict[str, DataLoader]:
        """Get combined DataLoaders for centralized baseline."""
        features, labels = self._load_or_generate()

        indices = np.arange(len(features))
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=self.val_ratio + self.test_ratio,
            random_state=self.random_seed,
            stratify=labels,
        )
        relative_test = self.test_ratio / (self.val_ratio + self.test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test,
            random_state=self.random_seed,
            stratify=labels[temp_idx],
        )

        train_features = self.scaler.fit_transform(features[train_idx])
        val_features = self.scaler.transform(features[val_idx])
        test_features = self.scaler.transform(features[test_idx])

        return {
            "train": DataLoader(
                NetworkTrafficDataset(train_features, labels[train_idx]),
                batch_size=self.batch_size, shuffle=True,
            ),
            "val": DataLoader(
                NetworkTrafficDataset(val_features, labels[val_idx]),
                batch_size=self.batch_size, shuffle=False,
            ),
            "test": DataLoader(
                NetworkTrafficDataset(test_features, labels[test_idx]),
                batch_size=self.batch_size, shuffle=False,
            ),
        }
