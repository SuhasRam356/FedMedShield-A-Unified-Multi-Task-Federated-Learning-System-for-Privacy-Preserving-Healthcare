"""
═══════════════════════════════════════════════════════════════
FedMedShield — FL Orchestrator
One-click script to run a full Federated Learning simulation.
Starts the server and N hospital clients via subprocesses.
Captures output and manages orderly shutdown.
═══════════════════════════════════════════════════════════════
"""

import subprocess
import time
import sys
import os
import argparse
import signal

processes = []

def cleanup(signum, frame):
    """Ensure all child processes are killed if script exits."""
    print("\n[Orchestrator] Terminating FL network...")
    for p in processes:
        try:
            p.terminate()
        except:
            pass
    sys.exit(0)

# Register cleanup for Ctrl+C
signal.signal(signal.SIGINT, cleanup)

def start_network(task: str, rounds: int, clients: int):
    print("═══════════════════════════════════════════════════════════════")
    print(f"  Starting FedMedShield Network for Task: {task.upper()}")
    print(f"  Hospitals: {clients} | FL Rounds: {rounds}")
    print("═══════════════════════════════════════════════════════════════")
    
    # Enable PYTHONPATH to include current dir
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    # 1. Start Server
    print("[Orchestrator] Booting Central Server...")
    server_cmd = [
        sys.executable, "fl_engine/server.py", 
        "--task", task, 
        "--rounds", str(rounds),
        "--clients", str(clients)
    ]
    server_process = subprocess.Popen(server_cmd, env=env)
    processes.append(server_process)

    # Wait for server to bind to port
    time.sleep(3)

    # 2. Start Clients
    for i in range(clients):
        hospital_id = f"hospital-{chr(ord('a') + i)}"
        print(f"[Orchestrator] Booting {hospital_id}...")
        
        client_cmd = [
            sys.executable, "fl_engine/client.py",
            "--id", hospital_id,
            "--server", "127.0.0.1:8080"
        ]
        
        client_process = subprocess.Popen(client_cmd, env=env)
        processes.append(client_process)
        
        # Slight stagger to prevent DB/IO congestion
        time.sleep(1)

    print("\n[Orchestrator] All nodes online. Federated Learning in progress...\n")

    # Wait for server to finish
    server_process.wait()
    print("\n[Orchestrator] Server completed training.")
    
    # Cleanup clients
    cleanup(None, None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedMedShield Orchestrator")
    parser.add_argument("--task", type=str, default="ehr", choices=["ehr", "imaging_tumor", "imaging_glaucoma", "drug", "ids"], help="Which AI module to train")
    parser.add_argument("--rounds", type=int, default=3, help="Number of FL rounds")
    parser.add_argument("--clients", type=int, default=4, help="Number of simulated hospitals")
    args = parser.parse_args()

    start_network(args.task, args.rounds, args.clients)
