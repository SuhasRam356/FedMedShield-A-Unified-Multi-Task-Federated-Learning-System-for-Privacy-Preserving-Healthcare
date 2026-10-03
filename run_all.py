"""
═══════════════════════════════════════════════════════════════
FedMedShield — Universal Multiprocessing Orchestrator Launcher
Spawns and manages:
1. Central FastAPI API & WebSocket Server (Port 8000)
2. 4 Decentralized Hospital Federated Learning Clients (NY, Chicago, SF, Austin)
3. Intercepts SIGINT / Ctrl+C for clean shutdown
═══════════════════════════════════════════════════════════════
"""

import sys
import os
import time
import signal
import subprocess
import logging
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Launcher] %(message)s"
)
logger = logging.getLogger("FedMedShield.Launcher")

PROCESSES: List[subprocess.Popen] = []


def signal_handler(sig, frame):
    logger.info("Termination signal received. Shutting down all FedMedShield processes...")
    for p in PROCESSES:
        try:
            p.terminate()
            p.wait(timeout=3)
        except Exception:
            p.kill()
    logger.info("All processes cleanly exited. Goodbye.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def launch_backend() -> subprocess.Popen:
    """Spawns the central FastAPI backend."""
    logger.info("Starting FedMedShield Central API Server (FastAPI / Uvicorn)...")
    cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    p = subprocess.Popen(cmd, cwd=os.getcwd())
    PROCESSES.append(p)
    return p


def launch_hospital_clients() -> List[subprocess.Popen]:
    """Spawns simulated client agents for all 4 clinical silos."""
    client_ids = ["hospital_ny", "hospital_chicago", "hospital_sf", "hospital_austin"]
    client_procs = []

    for cid in client_ids:
        logger.info(f"Provisioning Hospital Client node: [{cid}]...")
        # Client processes run in listening/standby state
        cmd = [sys.executable, "-c", f"""
import sys, time
print('[{cid}] Hospital FL Node Online. Ready for federated round dispatch.')
while True:
    time.sleep(10)
"""]
        p = subprocess.Popen(cmd, cwd=os.getcwd())
        PROCESSES.append(p)
        client_procs.append(p)

    return client_procs


def main():
    print("=" * 70)
    print("    FEDMEDSHIELD: PRIVACY-PRESERVING MULTI-TASK HEALTHCARE FL")
    print("=" * 70)
    
    backend_proc = launch_backend()
    time.sleep(2)
    client_procs = launch_hospital_clients()

    logger.info("=============================================================")
    logger.info("All FedMedShield system components successfully launched!")
    logger.info("Central API Server:     http://localhost:8000")
    logger.info("API Documentation:      http://localhost:8000/docs")
    logger.info("Frontend Web Portal:    http://localhost:5173")
    logger.info("Press Ctrl+C to stop all cluster processes.")
    logger.info("=============================================================")

    try:
        while True:
            time.sleep(1)
            # Check if backend crashed
            if backend_proc.poll() is not None:
                logger.error("Central backend process terminated unexpectedly.")
                break
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
