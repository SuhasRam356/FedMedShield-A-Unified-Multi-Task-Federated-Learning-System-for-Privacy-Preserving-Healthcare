"""
═══════════════════════════════════════════════════════════════
FedMedShield — FL Client Node
Represents a single hospital in the federated network.
Handles local training across all 4 modules (EHR, Imaging, Drug, IDS),
applies Differential Privacy, and masks updates for Secure Aggregation.
═══════════════════════════════════════════════════════════════
"""

import os
import time
import json
import torch
import flwr as fl
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np

# Data Loaders
from data.ehr_data.data_loader import EHRDataLoaderFactory
from data.imaging_data.data_loader import ImagingDataLoaderFactory
from data.drug_data.data_loader import DrugDataLoaderFactory
from data.network_data.data_loader import NetworkDataLoaderFactory

# Models & Training Scripts
from modules.module1_ehr.ehr_model import EHRMultiTaskModel
from modules.module1_ehr.train import train_ehr_model

from modules.module2_imaging.imaging_model import MedicalImagingModel, TumorDetector, GlaucomaDetector
from modules.module2_imaging.train import train_imaging_model

from modules.module3_drug.drug_model import DrugProteinBindingModel
from modules.module3_drug.train import train_drug_model

from modules.module4_ids.ids_model import IntrusionDetectionModel
from modules.module4_ids.train import train_ids_model

# Privacy
from privacy.differential_privacy import DifferentialPrivacyEngine
from privacy.secure_aggregation import SecureAggregationEngine

logger = logging.getLogger(__name__)


def safe_load_weights(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """Always validate weight shapes before aggregation/loading into model."""
    current_state = model.state_dict()
    params_dict = zip(current_state.keys(), parameters)
    state_dict = OrderedDict()
    for k, v in params_dict:
        expected_shape = current_state[k].shape
        received_tensor = torch.tensor(v)
        received_shape = received_tensor.shape
        if expected_shape == received_shape:  # ✅ Shape check
            state_dict[k] = received_tensor
        else:
            logger.warning(f"⚠️ Shape mismatch at {k}: expected {expected_shape}, got {received_shape}")
            state_dict[k] = current_state[k]  # Keep old weight
    model.load_state_dict(state_dict, strict=False)


class HospitalClient(fl.client.NumPyClient):
    """
    Flower Client representing a hospital node.
    It can train any of the 4 sub-models depending on server instructions.
    """

    def __init__(
        self,
        client_id: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        auth_token: Optional[str] = None,
        dp_level: str = "client_level",
    ):
        self.client_id = client_id
        self.device = torch.device(device)
        self.auth_token = auth_token or os.getenv("FL_CLIENT_TOKEN")
        
        logger.info(f"Initializing HospitalClient {client_id} on {self.device}...")
        
        # Privacy Engines
        self.dp_engine = DifferentialPrivacyEngine(
            target_epsilon=5.0, 
            noise_multiplier=1.0, 
            initial_max_norm=1.0,
            dp_level=dp_level,
        )
        self.secagg_engine = SecureAggregationEngine(client_id=client_id)
        
        # We will lazy-load data and models when the server requests a specific task
        self.models = {}
        self.dataloaders = {}
        
        # Mapping from string identifiers to model classes
        self.model_registry = {
            "ehr": EHRMultiTaskModel,
            "imaging": TumorDetector,
            "imaging_tumor": TumorDetector,
            "imaging_glaucoma": GlaucomaDetector,
            "drug": DrugProteinBindingModel,
            "ids": IntrusionDetectionModel,
        }

    def _get_model(self, task_name: str) -> torch.nn.Module:
        """Lazy initialization of models."""
        if task_name not in self.models:
            logger.info(f"[{self.client_id}] Initializing model for task: {task_name}")
            ModelClass = self.model_registry[task_name]
            self.models[task_name] = ModelClass().to(self.device)
        return self.models[task_name]

    def _get_dataloaders(self, task_name: str) -> Dict:
        """Lazy initialization of data loaders."""
        if task_name not in self.dataloaders:
            logger.info(f"[{self.client_id}] Loading data for task: {task_name}")
            
            if task_name == "ehr":
                factory = EHRDataLoaderFactory()
                self.dataloaders[task_name] = factory.get_hospital_dataloaders(self.client_id)
            elif task_name.startswith("imaging") or task_name == "imaging":
                factory = ImagingDataLoaderFactory()
                img_task = "glaucoma" if "glaucoma" in task_name else ("covid_xray" if "covid" in task_name else "tumor")
                self.dataloaders[task_name] = factory.get_hospital_dataloaders(self.client_id, task=img_task)
            elif task_name == "drug":
                factory = DrugDataLoaderFactory()
                self.dataloaders[task_name] = factory.get_hospital_dataloaders(self.client_id)
            elif task_name == "ids":
                factory = NetworkDataLoaderFactory()
                self.dataloaders[task_name] = factory.get_hospital_dataloaders(self.client_id)
                
        return self.dataloaders[task_name]

    def set_parameters(self, task_name: str, parameters: List[np.ndarray]):
        """Load weights from server into local model with shape validation."""
        model = self._get_model(task_name)
        safe_load_weights(model, parameters)

    def get_parameters(self, task_name: str, config: Dict) -> List[np.ndarray]:
        """Extract weights from local model to send to server."""
        model = self._get_model(task_name)
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def fit(self, parameters: List[np.ndarray], config: Dict):
        """
        Train the model locally.
        The server dictates which task to train via the config dictionary.
        """
        task_name = config.get("task", "ehr")
        epochs = config.get("epochs", 1)
        fedprox_mu = config.get("fedprox_mu", 0.0)
        use_dp = config.get("use_dp", True)
        use_secagg = config.get("use_secagg", True)
        
        logger.info(f"[{self.client_id}] FIT starting. Task: {task_name}, Epochs: {epochs}")

        model = self._get_model(task_name)

        # Assert parameter count and shapes before the first round
        round_num = config.get("round", config.get("current_round", 1))
        if round_num == 1:
            current_state = model.state_dict()
            assert len(parameters) == len(current_state), (
                f"Round 1 param count mismatch for '{task_name}': "
                f"expected {len(current_state)}, received {len(parameters)}"
            )
            for (p_name, tensor_val), arr_val in zip(current_state.items(), parameters):
                assert tensor_val.shape == arr_val.shape, (
                    f"Round 1 param shape mismatch for '{task_name}' on layer '{p_name}': "
                    f"expected {tensor_val.shape}, received {arr_val.shape}"
                )

        # 1. Update local model with global parameters
        self.set_parameters(task_name, parameters)
        
        # Keep a copy of the global model for FedProx and DP pseudo-gradients
        global_model = self.model_registry[task_name]().to(self.device)
        global_model.load_state_dict(model.state_dict())
        
        # 2. Get local data
        loaders = self._get_dataloaders(task_name)
        train_loader = loaders["train"]
        val_loader = loaders["val"]
        
        # 3. Train the model using the appropriate training script
        results = None
        if task_name == "ehr":
            results = train_ehr_model(
                model, train_loader, val_loader, self.device, 
                num_epochs=epochs, global_model=global_model, fedprox_mu=fedprox_mu
            )
        elif task_name.startswith("imaging"):
            results = train_imaging_model(
                model, train_loader, val_loader, self.device, 
                num_epochs=epochs, global_model=global_model, fedprox_mu=fedprox_mu
            )
        elif task_name == "drug":
            results = train_drug_model(
                model, train_loader, val_loader, self.device, 
                num_epochs=epochs, global_model=global_model, fedprox_mu=fedprox_mu
            )
        elif task_name == "ids":
            results = train_ids_model(
                model, train_loader, val_loader, self.device, 
                num_epochs=epochs, global_model=global_model, fedprox_mu=fedprox_mu
            )

        # Participant Authentication Check (if token required)
        expected_token = config.get("auth_token", "")
        if expected_token and self.auth_token and self.auth_token != expected_token:
            logger.error(f"[{self.client_id}] Participant authentication failed: invalid token")
            raise PermissionError(f"Client {self.client_id} unauthorized")

        # Ensure model has the best weights from training
        model.load_state_dict(results["model_state_dict"])
        final_state = model.state_dict()
        
        # 4. Apply Differential Privacy
        metrics = {}
        if use_dp:
            logger.info(f"[{self.client_id}] Applying Differential Privacy...")
            dp_level = config.get("dp_level", self.dp_engine.dp_level)
            self.dp_engine.dp_level = dp_level
            
            batch_size = getattr(train_loader, "batch_size", 32)
            total_samples = len(train_loader.dataset) if hasattr(train_loader, "dataset") else 1000
            
            if dp_level == "client_level":
                eff_sample_rate = float(config.get("num_clients", 2) / max(1, config.get("total_hospitals", 4)))
            else:
                eff_sample_rate = float(batch_size / max(1, total_samples))

            final_state, dp_metrics = self.dp_engine.apply_dp(
                local_model_state=final_state,
                global_model_state=global_model.state_dict(),
                local_steps=epochs * len(train_loader),
                sample_rate=eff_sample_rate
            )
            metrics.update(dp_metrics)
            
        # 5. Apply Secure Aggregation Masking with Key Rotation
        if use_secagg:
            logger.info(f"[{self.client_id}] Applying Secure Aggregation Masking...")
            fresh_pub_key = self.secagg_engine.rotate_keys()
            metrics["secagg_public_key"] = fresh_pub_key

            active_clients_val = config.get("active_clients", "")
            peer_keys_val = config.get("peer_keys", "")

            try:
                active_clients = json.loads(active_clients_val) if isinstance(active_clients_val, str) and active_clients_val else [self.client_id]
            except Exception:
                active_clients = [self.client_id]

            try:
                peer_keys = json.loads(peer_keys_val) if isinstance(peer_keys_val, str) and peer_keys_val else {}
            except Exception:
                peer_keys = {}
            
            self.secagg_engine.receive_public_keys(peer_keys)
            final_state = self.secagg_engine.mask_local_update(final_state)

        # 6. Extract numpy arrays to return to server
        # We must return them in the exact order as model.state_dict()
        updated_params = [val.cpu().numpy() for _, val in final_state.items()]
        num_samples = len(train_loader.dataset)
        
        # Suppress raw update norms, clipping bounds, and patient diagnostics from server metrics
        sanitized_metrics = {
            "loss": float(results.get("val_loss", 0.0)),
            "accuracy": float(results.get("val_acc", 0.0)),
            "current_epsilon": float(metrics.get("current_epsilon", 0.0)),
            "target_epsilon": float(metrics.get("target_epsilon", 5.0)),
            "budget_exhausted": bool(metrics.get("budget_exhausted", False)),
            "dp_level": str(self.dp_engine.dp_level),
        }
        if "secagg_public_key" in metrics:
            sanitized_metrics["secagg_public_key"] = metrics["secagg_public_key"]

        logger.info(f"[{self.client_id}] FIT complete. Sending {len(updated_params)} tensors.")
        
        return updated_params, num_samples, sanitized_metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict):
        """Evaluate the global model on local validation data."""
        task_name = config.get("task", "ehr")
        logger.info(f"[{self.client_id}] EVALUATE starting. Task: {task_name}")
        
        self.set_parameters(task_name, parameters)
        model = self._get_model(task_name)
        
        loaders = self._get_dataloaders(task_name)
        test_loader = loaders["test"]
        
        # We use the evaluation functions defined in the train scripts
        if task_name == "ehr":
            from modules.module1_ehr.ehr_model import EHRMultiTaskLoss
            from modules.module1_ehr.train import evaluate as eval_ehr
            metrics = eval_ehr(model, test_loader, EHRMultiTaskLoss().to(self.device), self.device)
            loss = metrics.pop("total_loss")
            
        elif task_name.startswith("imaging"):
            from modules.module2_imaging.train import evaluate as eval_img
            metrics = eval_img(model, test_loader, torch.nn.CrossEntropyLoss(), self.device)
            loss = metrics.pop("loss")
            
        elif task_name == "drug":
            from modules.module3_drug.drug_model import DrugBindingLoss
            from modules.module3_drug.train import evaluate as eval_drug
            metrics = eval_drug(model, test_loader, DrugBindingLoss().to(self.device), self.device)
            loss = metrics.pop("total_loss")
            
        elif task_name == "ids":
            from modules.module4_ids.train import evaluate as eval_ids
            metrics = eval_ids(model, test_loader, torch.nn.CrossEntropyLoss(), self.device)
            loss = metrics.pop("loss")
            
        num_samples = len(test_loader.dataset)
        logger.info(f"[{self.client_id}] EVALUATE complete. Loss: {loss:.4f}")
        
        return float(loss), num_samples, metrics


# Exported alias
FLClient = HospitalClient



def load_root_certificate(ca_path: Optional[str] = None) -> Optional[bytes]:
    """Loads root certificate for TLS Flower client if provided."""
    path = ca_path or os.getenv("FL_CA_CERT")
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            logger.info("Loaded root CA certificate for encrypted Flower client.")
            return f.read()
    return None


def start_client(
    client_id: str,
    server_address: str = "127.0.0.1:8080",
    max_retries: int = 5,
    retry_delay: int = 2,
    ca_cert: Optional[str] = None
):
    """
    Entry point to start the flower client with optional TLS/mTLS encryption.
    Normalizes localhost to 127.0.0.1 to avoid Windows IPv6 resolution issues,
    and adds retry logic to wait for the FL server if not yet ready.
    """
    # Normalize localhost / 0.0.0.0 to 127.0.0.1 to prevent Windows IPv6 [::1]:8080 connection failures
    if "localhost:" in server_address:
        server_address = server_address.replace("localhost:", "127.0.0.1:")
    elif "0.0.0.0:" in server_address:
        server_address = server_address.replace("0.0.0.0:", "127.0.0.1:")

    client = HospitalClient(client_id=client_id)
    root_cert = load_root_certificate(ca_cert)
    if root_cert:
        logger.info(f"Connecting to encrypted TLS FL server at {server_address}...")
    else:
        logger.warning(f"Connecting to FL server at {server_address} in UNENCRYPTED PLAINTEXT mode.")
    
    for attempt in range(1, max_retries + 1):
        try:
            fl.client.start_numpy_client(
                server_address=server_address,
                client=client,
                root_certificates=root_cert
            )
            break
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    f"Connection attempt {attempt}/{max_retries} to {server_address} failed ({e}). "
                    f"Waiting {retry_delay}s for server..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to FL server at {server_address} after {max_retries} attempts.")
                raise e


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="FedMedShield Client")
    parser.add_argument("--id", type=str, required=True, help="Client ID (e.g., hospital-a)")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Server address (default: 127.0.0.1:8080)")
    parser.add_argument("--retries", type=int, default=5, help="Max connection retries")
    parser.add_argument("--ca-cert", type=str, default=None, help="Path to CA certificate for TLS")
    args = parser.parse_args()
    
    start_client(args.id, args.server, max_retries=args.retries, ca_cert=args.ca_cert)
