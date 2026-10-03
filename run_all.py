"""
═══════════════════════════════════════════════════════════════
FedMedShield — Universal Multiprocessing Orchestrator Launcher
Spawns and manages:
1. FL Server (Port 8080)
2. Central FastAPI API & WebSocket Server (Port 8000)
3. 4 Decentralized Hospital Federated Learning Clients (A, B, C, D)
4. Intercepts SIGINT / Ctrl+C for clean shutdown
═══════════════════════════════════════════════════════════════
"""

import multiprocessing
import subprocess
import sys
import time
import os
import signal
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(name)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger("FedMedShield")

processes = []  # Global list for cleanup


def run_process(name: str, command: list, env_vars: dict = {}):
    """Run a subprocess with error handling and real-time streaming."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    env.update(env_vars)
    
    logger.info(f"🚀 Starting {name}...")
    
    try:
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        # Stream output in real-time
        for line in iter(proc.stdout.readline, ''):
            print(f"[{name}] {line}", end='')
        proc.wait()
        
        if proc.returncode != 0:
            logger.error(f"❌ {name} crashed with code {proc.returncode}")
        
    except Exception as e:
        logger.error(f"❌ {name} failed to start: {e}")


def signal_handler(sig, frame):
    """Graceful shutdown for all processes."""
    logger.info("\n🛑 Shutting down all FedMedShield services...")
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
    logger.info("✅ All services stopped. Goodbye!")
    sys.exit(0)


if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    service_configs = [
        {
            "name": "FL-Server",
            "command": [sys.executable, "fl_engine/server.py"],
            "env": {"FL_PORT": "8080"},
            "delay": 0
        },
        {
            "name": "API-Server",
            "command": [sys.executable, "-m", "uvicorn",
                       "backend.main:app", "--host", "0.0.0.0",
                       "--port", "8000"],
            "env": {},
            "delay": 3  # Wait for FL server
        },
        {
            "name": "Hospital-A",
            "command": [sys.executable, "clients/hospital_client.py"],
            "env": {"HOSPITAL_ID": "Hospital-A",
                   "HOSPITAL_PORT": "9001",
                   "MODULES": "ehr,sepsis,covid",
                   "FL_SERVER": "127.0.0.1:8080"},
            "delay": 5  # Wait for server + API
        },
        {
            "name": "Hospital-B",
            "command": [sys.executable, "clients/hospital_client.py"],
            "env": {"HOSPITAL_ID": "Hospital-B",
                   "HOSPITAL_PORT": "9002",
                   "MODULES": "imaging,tumor,glaucoma",
                   "FL_SERVER": "127.0.0.1:8080"},
            "delay": 6
        },
        {
            "name": "Hospital-C",
            "command": [sys.executable, "clients/hospital_client.py"],
            "env": {"HOSPITAL_ID": "Hospital-C",
                   "HOSPITAL_PORT": "9003",
                   "MODULES": "ehr,imaging,drug",
                   "FL_SERVER": "127.0.0.1:8080"},
            "delay": 7
        },
        {
            "name": "Hospital-D",
            "command": [sys.executable, "clients/hospital_client.py"],
            "env": {"HOSPITAL_ID": "Hospital-D",
                   "HOSPITAL_PORT": "9004",
                   "MODULES": "ids,ehr",
                   "FL_SERVER": "127.0.0.1:8080"},
            "delay": 8
        },
    ]

    # Start all services
    for config in service_configs:
        time.sleep(config["delay"])
        p = multiprocessing.Process(
            target=run_process,
            args=(config["name"], config["command"], config["env"]),
            name=config["name"]
        )
        p.start()
        processes.append(p)
        logger.info(f"✅ {config['name']} started (PID: {p.pid})")

    print("\n" + "="*60)
    print("  🏥 FedMedShield — ALL SERVICES RUNNING")
    print("="*60)
    print("  🎨 Frontend  → http://localhost:5173")
    print("  ⚙️  API Docs  → http://localhost:8000/docs")
    print("  🌐 FL Server → localhost:8080")
    print("  🏥 Hospital A → localhost:9001")
    print("  🏥 Hospital B → localhost:9002")
    print("  🏥 Hospital C → localhost:9003")
    print("  🏥 Hospital D → localhost:9004")
    print("="*60)
    print("  Press Ctrl+C to stop all services")
    print("="*60 + "\n")

    # Keep main process alive
    for p in processes:
        p.join()
