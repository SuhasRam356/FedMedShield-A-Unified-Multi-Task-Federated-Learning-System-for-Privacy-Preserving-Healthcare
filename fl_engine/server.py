"""
═══════════════════════════════════════════════════════════════
FedMedShield — FL Server Node
Runs the Flower server with a custom Federated Averaging strategy.
Handles Secure Aggregation unmasking, global model saving, and
metric aggregation across hospitals.
═══════════════════════════════════════════════════════════════
"""

import os
import time
import torch
import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays
)
from typing import Dict, List, Tuple, Optional, Union
import logging
import numpy as np

from privacy.secure_aggregation import SecureAggregationEngine

logger = logging.getLogger(__name__)


class SecureFedAvgStrategy(fl.server.strategy.FedAvg):
    """
    Custom FL Strategy extending FedAvg.
    Adds support for:
    1. Secure Aggregation (unmasking client updates securely).
    2. Dynamic client configuration (sending public keys to clients).
    3. Custom metric aggregation (e.g., F1 scores, DP Epsilon tracking).
    4. Global model checkpointing.
    """

    def __init__(
        self,
        task_name: str,
        use_secagg: bool = True,
        use_dp: bool = True,
        num_hospitals: int = 4,
        save_dir: str = "checkpoints",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.task_name = task_name
        self.use_secagg = use_secagg
        self.use_dp = use_dp
        self.num_hospitals = num_hospitals
        self.save_dir = save_dir
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Simulate knowing client public keys for SecAgg setup
        # In a real distributed setup, clients would send these via a secure side-channel
        # or in a setup phase prior to training.
        from privacy.secure_aggregation import KeyGenerator
        self.client_ids = [f"hospital-{chr(ord('a') + i)}" for i in range(num_hospitals)]
        self.public_keys = {
            cid: KeyGenerator.generate_keypair(cid)[1] for cid in self.client_ids
        }

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        
        # Select clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom config for this round
        fit_config = {
            "task": self.task_name,
            "epochs": 1,
            "fedprox_mu": 0.1,  # Non-IID handling
            "use_dp": self.use_dp,
            "use_secagg": self.use_secagg,
            "active_clients": self.client_ids,
            "peer_keys": self.public_keys,
            "server_round": server_round,
        }

        fit_ins = FitIns(parameters, fit_config)
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        
        if self.fraction_evaluate == 0.0:
            return []
            
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        eval_config = {
            "task": self.task_name,
            "server_round": server_round,
        }
        
        eval_ins = EvaluateIns(parameters, eval_config)
        return [(client, eval_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using Secure Aggregation."""
        if not results:
            return None, {}
            
        logger.info(f"[Round {server_round}] Aggregating {len(results)} client updates...")

        # Extract DP metrics to monitor privacy budget across clients
        dp_metrics = [res.metrics for _, res in results if "current_epsilon" in res.metrics]
        if dp_metrics:
            max_eps = max([m["current_epsilon"] for m in dp_metrics])
            avg_eps = sum([m["current_epsilon"] for m in dp_metrics]) / len(dp_metrics)
            logger.info(f"Privacy Budget Tracking - Max Epsilon: {max_eps:.2f}, Avg Epsilon: {avg_eps:.2f}")

        # Extract weights and number of samples
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        aggregated_ndarrays = None

        if self.use_secagg:
            # --- SECURE AGGREGATION LOGIC ---
            logger.info("Performing Secure Aggregation (Unmasking + FedAvg)...")
            
            # Convert List[np.ndarray] back to dictionary format required by SecAgg engine
            # We don't have the original keys, so we use dummy keys 'layer_0', 'layer_1', etc.
            masked_dicts = []
            sample_counts = []
            
            for client_ndarrays, num_samples in weights_results:
                state_dict = {
                    f"layer_{i}": torch.tensor(arr) 
                    for i, arr in enumerate(client_ndarrays)
                }
                masked_dicts.append(state_dict)
                sample_counts.append(float(num_samples))
                
            # Perform server-side unmasking and aggregation
            agg_dict = SecureAggregationEngine.aggregate_masked_updates(
                masked_updates=masked_dicts,
                weights=sample_counts
            )
            
            # Convert back to List[np.ndarray]
            aggregated_ndarrays = [
                agg_dict[f"layer_{i}"].cpu().numpy() 
                for i in range(len(agg_dict))
            ]
        else:
            # --- STANDARD FEDAVG LOGIC ---
            from flwr.server.strategy.aggregate import aggregate
            aggregated_ndarrays = aggregate(weights_results)

        if aggregated_ndarrays is None:
            return None, {}

        # Save global model checkpoint
        if aggregated_ndarrays is not None:
            save_path = os.path.join(self.save_dir, f"{self.task_name}_round_{server_round}.npz")
            np.savez(save_path, *aggregated_ndarrays)
            logger.info(f"Saved global model checkpoint to {save_path}")

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        return parameters_aggregated, {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics (Accuracies, F1 scores, etc.)."""
        if not results:
            return None, {}

        # Aggregate Loss (Weighted average based on samples)
        loss_aggregated = sum(
            [res.loss * res.num_examples for _, res in results]
        ) / sum([res.num_examples for _, res in results])

        # Aggregate custom metrics (Accuracies, etc.)
        metrics_aggregated = {}
        total_examples = sum([res.num_examples for _, res in results])

        # Find all metric keys
        metric_keys = set()
        for _, res in results:
            metric_keys.update(res.metrics.keys())

        for key in metric_keys:
            # Weighted average for metrics
            metric_sum = sum(
                [res.metrics.get(key, 0.0) * res.num_examples for _, res in results]
            )
            metrics_aggregated[key] = metric_sum / total_examples

        logger.info(f"[Round {server_round}] EVALUATION COMPLETE:")
        logger.info(f"  Loss: {loss_aggregated:.4f}")
        for key, val in metrics_aggregated.items():
            if "acc" in key or "f1" in key or "rate" in key:
                logger.info(f"  {key}: {val:.4f}")

        return loss_aggregated, metrics_aggregated


def start_server(
    task_name: str = "ehr",
    num_rounds: int = 20,
    num_hospitals: int = 4,
    min_clients: int = 2,
    round_timeout: int = 300,
    use_secagg: bool = True,
    use_dp: bool = True,
    server_address: str = "0.0.0.0:8080",
    strategy: Optional[fl.server.strategy.Strategy] = None,
):
    """
    Entry point to start the FL server.
    Adds wait_for_clients delay and round_timeout to prevent premature start and client connection drops.
    """
    logger.info(f"Starting FL Server for task: {task_name}")
    logger.info(f"Configuration -> Host: {server_address} (0.0.0.0 NOT localhost), Rounds: {num_rounds}, Timeout: {round_timeout}s, Min Clients: {min_clients}")
    logger.info(f"Privacy Config -> DP: {use_dp}, SecAgg: {use_secagg}")

    # Wait for clients to register and sockets to stabilize
    logger.info("Waiting 2s for clients to register...")
    time.sleep(2)  # Wait for clients to register

    # We need to initialize the global model with random weights to send to clients in Round 1
    if strategy is None:
        from flwr.common import ndarrays_to_parameters
        
        # Import the correct model based on task
        if task_name == "ehr":
            from modules.module1_ehr.ehr_model import EHRMultiTaskModel
            initial_model = EHRMultiTaskModel()
        elif task_name.startswith("imaging"):
            from modules.module2_imaging.imaging_model import TumorDetector
            initial_model = TumorDetector()  # Standardizes on 3-class for init
        elif task_name == "drug":
            from modules.module3_drug.drug_model import DrugProteinBindingModel
            initial_model = DrugProteinBindingModel()
        elif task_name == "ids":
            from modules.module4_ids.ids_model import IntrusionDetectionModel
            initial_model = IntrusionDetectionModel()
        else:
            raise ValueError(f"Unknown task: {task_name}")
            
        initial_weights = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
        initial_parameters = ndarrays_to_parameters(initial_weights)

        # Configure Strategy with lowered client thresholds so training succeeds even if clients fail
        effective_min_clients = min(min_clients, num_hospitals)
        strategy = SecureFedAvgStrategy(
            task_name=task_name,
            use_secagg=use_secagg,
            use_dp=use_dp,
            num_hospitals=num_hospitals,
            fraction_fit=1.0,  # Train on all clients
            fraction_evaluate=1.0,  # Eval on all clients
            min_fit_clients=effective_min_clients,          # ⬅️ Lower this if clients fail
            min_available_clients=effective_min_clients,    # ⬅️ Lower this too
            min_evaluate_clients=effective_min_clients,
            initial_parameters=initial_parameters,
        )

def load_tls_certificates(
    ca_cert: Optional[str] = None,
    server_cert: Optional[str] = None,
    server_key: Optional[str] = None
) -> Optional[Tuple[bytes, bytes, bytes]]:
    """Loads TLS certificates from file paths or environment variables if present."""
    ca_path = ca_cert or os.getenv("FL_CA_CERT")
    cert_path = server_cert or os.getenv("FL_SERVER_CERT")
    key_path = server_key or os.getenv("FL_SERVER_KEY")

    if cert_path and key_path and os.path.exists(cert_path) and os.path.exists(key_path):
        with open(cert_path, "rb") as f:
            cert_bytes = f.read()
        with open(key_path, "rb") as f:
            key_bytes = f.read()
        ca_bytes = b""
        if ca_path and os.path.exists(ca_path):
            with open(ca_path, "rb") as f:
                ca_bytes = f.read()
        logger.info("Loaded TLS certificates for encrypted Flower server.")
        return (ca_bytes, cert_bytes, key_bytes)
    return None


def start_server(
    task_name: str = "ehr",
    num_rounds: int = 20,
    num_hospitals: int = 4,
    min_clients: int = 2,
    round_timeout: int = 300,
    use_secagg: bool = True,
    use_dp: bool = True,
    strategy: Optional[fl.server.strategy.Strategy] = None,
    server_address: str = "0.0.0.0:8080",
    ca_cert: Optional[str] = None,
    server_cert: Optional[str] = None,
    server_key: Optional[str] = None
):
    """
    Launches the Flower Federated Learning Server with optional TLS/mTLS encryption.
    """
    if strategy is None:
        # Import the correct model based on task
        if task_name == "ehr":
            from modules.module1_ehr.ehr_model import EHRMultiTaskModel
            initial_model = EHRMultiTaskModel()
        elif task_name.startswith("imaging"):
            from modules.module2_imaging.imaging_model import TumorDetector
            initial_model = TumorDetector()  # Standardizes on 3-class for init
        elif task_name == "drug":
            from modules.module3_drug.drug_model import DrugProteinBindingModel
            initial_model = DrugProteinBindingModel()
        elif task_name == "ids":
            from modules.module4_ids.ids_model import IntrusionDetectionModel
            initial_model = IntrusionDetectionModel()
        else:
            raise ValueError(f"Unknown task: {task_name}")
            
        initial_weights = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
        initial_parameters = ndarrays_to_parameters(initial_weights)

        # Configure Strategy with lowered client thresholds so training succeeds even if clients fail
        effective_min_clients = min(min_clients, num_hospitals)
        strategy = SecureFedAvgStrategy(
            task_name=task_name,
            use_secagg=use_secagg,
            use_dp=use_dp,
            num_hospitals=num_hospitals,
            fraction_fit=1.0,  # Train on all clients
            fraction_evaluate=1.0,  # Eval on all clients
            min_fit_clients=effective_min_clients,          # ⬅️ Lower this if clients fail
            min_available_clients=effective_min_clients,    # ⬅️ Lower this too
            min_evaluate_clients=effective_min_clients,
            initial_parameters=initial_parameters,
        )

    certs = load_tls_certificates(ca_cert=ca_cert, server_cert=server_cert, server_key=server_key)
    if certs:
        logger.info(f"Binding encrypted TLS FL server to {server_address}...")
    else:
        logger.warning(
            "FL server starting in UNENCRYPTED PLAINTEXT mode. "
            "For production clinical federation, configure TLS/mTLS via --server-cert and --server-key or FL_SERVER_CERT and FL_SERVER_KEY."
        )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(
            num_rounds=num_rounds,
            round_timeout=round_timeout
        ),
        strategy=strategy,
        certificates=certs
    )


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="FedMedShield Server")
    parser.add_argument("--task", type=str, default="ehr", help="Task to train (ehr, imaging_tumor, drug, ids)")
    parser.add_argument("--rounds", type=int, default=20, help="Number of FL rounds (default: 20)")
    parser.add_argument("--clients", type=int, default=4, help="Number of hospitals (default: 4)")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum clients required (default: 2)")
    parser.add_argument("--timeout", type=int, default=300, help="Round timeout in seconds (default: 300)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host IP (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--no-secagg", action="store_true", help="Disable Secure Aggregation")
    parser.add_argument("--no-dp", action="store_true", help="Disable Differential Privacy")
    parser.add_argument("--ca-cert", type=str, default=None, help="Path to CA certificate for mTLS")
    parser.add_argument("--server-cert", type=str, default=None, help="Path to server certificate")
    parser.add_argument("--server-key", type=str, default=None, help="Path to server private key")
    args = parser.parse_args()
    
    server_addr = f"{args.host}:{args.port}"
    start_server(
        task_name=args.task,
        num_rounds=args.rounds,
        num_hospitals=args.clients,
        min_clients=args.min_clients,
        round_timeout=args.timeout,
        use_secagg=not args.no_secagg,
        use_dp=not args.no_dp,
        server_address=server_addr,
        ca_cert=args.ca_cert,
        server_cert=args.server_cert,
        server_key=args.server_key,
    )
