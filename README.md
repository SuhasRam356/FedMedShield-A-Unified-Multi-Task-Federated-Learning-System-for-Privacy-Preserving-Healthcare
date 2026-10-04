<div align="center">

# 🏥 FedMedShield

### A Unified Multi-Task Federated Learning System for Privacy-Preserving Healthcare

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.3-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)
![Flower](https://img.shields.io/badge/Flower_FL-1.11-FF6F61?style=for-the-badge)
![License](https://img.shields.io/badge/License-IEEE_Research-blue?style=for-the-badge)

*Prepared for IEEE Panel Evaluation — Federated Learning in Healthcare Systems*

---

**A research simulation exploring multi-task federated learning workflows across distributed medical nodes.**

FedMedShield investigates federated learning architectures across simulated healthcare scenarios (clinical tabular EHR, ResNet imaging, bioactivity screening, and network security). In federated training mode, participating nodes train locally on client-side datasets and transmit only weight updates.

> [!IMPORTANT]
> **Privacy & Regulatory Scope:** FedMedShield is an academic research prototype and does **not** establish legal HIPAA or GDPR compliance, nor does it certify zero-leakage security. Model weights and metadata are not inherently private without strict institutional governance. Furthermore, while Flower federated training operates on local client loaders, the interactive web demonstration UI inference paths (such as the disease prediction and drug screening dashboards) transmit sample inputs directly to the central demonstration API host for evaluation.

</div>

---

## 📑 Table of Contents

| Section | Description |
|---------|-------------|
| [Problem Statement](#-problem-statement) | Why this project exists |
| [System Architecture](#-system-architecture) | How everything connects |
| [Project Structure](#-full-project-structure) | Every file and folder explained |
| [Phase 1 — Project Setup](#-phase-1--project-setup) | Environment, dependencies, configs |
| [Phase 2 — Type Definitions](#-phase-2--type-definitions-frontend) | TypeScript contracts for the frontend |
| [Phase 3 — Data Generation](#-phase-3--synthetic-data-generation) | Creating realistic medical data |
| [Phase 4 — AI Model Architectures](#-phase-4--ai-model-architectures-pytorch) | The 4 neural network modules |
| [Phase 5 — Privacy Engines](#-phase-5--privacy-engines) | Differential Privacy + Secure Aggregation |
| [Phase 6 — Federated Learning Engine](#-phase-6--federated-learning-engine) | Server, Client, Strategies, Orchestrator |
| [Phase 7 — Backend API](#-phase-7--backend-api-fastapi) | REST endpoints, WebSockets, databases |
| [Phase 8 — Frontend Dashboard](#-phase-8--frontend-dashboard-react--vite) | UI pages, charts, cards, forms |
| [Phase 9 — Evaluation & IEEE Plots](#-phase-9--evaluation--ieee-publication-plots) | ROC curves, confusion matrices, benchmarks |
| [Phase 10 — Deployment & Launch](#-phase-10--deployment--launch) | PM2, startup scripts, environment |
| [Quick Start](#-quick-start-guide) | Get running in 2 minutes |
| [API Reference](#-api-reference) | All backend endpoints |
| [Tech Stack](#-complete-tech-stack) | Every library and framework used |
| [Research References](#-research-references) | Papers and citations |
| [License](#-license) | Terms of use |

---

## 🎯 Problem Statement

Hospitals around the world face a paradox:

> **To build better AI for diagnosing diseases, they need more patient data.
> But to protect patients, they cannot share that data with anyone.**

Traditional machine learning requires centralizing all data in one place — which is illegal under HIPAA (USA), GDPR (Europe), and other privacy regulations. This means smaller hospitals are stuck with small datasets and weaker models.

**FedMedShield's Solution:** Instead of bringing data to the model, we bring the model to the data.

```
┌──────────────────────────────────────────────────────────────────────┐
│                     TRADITIONAL APPROACH (Broken)                   │
│                                                                      │
│  Hospital A ──────┐                                                  │
│  Hospital B ──────┼── All Data ──► Central Server ──► Train Model    │
│  Hospital C ──────┘     ❌ Privacy violation!                        │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    FEDMEDSHIELD APPROACH (Secure)                    │
│                                                                      │
│  Hospital A ── Train locally ── Send model weights (not data) ──┐   │
│  Hospital B ── Train locally ── Send model weights (not data) ──┼─► │
│  Hospital C ── Train locally ── Send model weights (not data) ──┘   │
│                                                      │               │
│                              Aggregate ◄─────────────┘               │
│                                  │                                   │
│                          Global Model ✅                             │
│                     (No patient data ever left!)                     │
└──────────────────────────────────────────────────────────────────────┘
```

In federated training mode:
1. The client downloads current global parameters from the Flower server
2. Trains locally using its node-specific dataset loaders
3. Computes updated model parameters
4. The server aggregates client updates into a revised global model checkpoint
5. The process repeats over configured federation rounds

The framework incorporates **reviewed privacy mechanisms**:
- **Differential Privacy (DP Engine):** Validated Rényi Differential Privacy (RDP) using Opacus `RDPAccountant`, explicit client/patient-level cohort calibration, pre-release budget gating, and metadata bypass prevention.
- **Secure Aggregation (SecAgg+ Engine):** Upstream Flower SecAgg+ reviewed primitives with X25519 ECDH key exchange, forward-secret key rotation, Shamir $t$-out-of-$U$ secret sharing for dropout tolerance, and stochastic quantization.

---

## 🏗 System Architecture

```
                            ╔══════════════════════════════╗
                            ║      FEDMEDSHIELD v2.0       ║
                            ║  System Architecture Diagram  ║
                            ╚══════════════════════════════╝

    ┌─────────────────────────────────────────────────────────────────────┐
    │                          FRONTEND LAYER                            │
    │                     (React 18 + Vite + Tailwind)                   │
    │                                                                     │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
    │  │Dashboard │ │ Network  │ │ Privacy  │ │  Nodes   │ │ Settings │ │
    │  │(Realtime)│ │  Map     │ │ Report   │ │ Manager  │ │  Panel   │ │
    │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │
    │       │             │            │             │            │       │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
    │  │Medical   │ │  Drug    │ │Intrusion │ │ Disease  │              │
    │  │Imaging   │ │Discovery │ │Detection │ │Prediction│              │
    │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘              │
    │       └──────────┬──┴────────────┴──────┬─────┘                    │
    │                  │    Axios + WebSocket  │                          │
    └──────────────────┼──────────────────────┼──────────────────────────┘
                       │                      │
    ┌──────────────────┼──────────────────────┼──────────────────────────┐
    │                  ▼   BACKEND API LAYER  ▼                          │
    │                  (FastAPI + Uvicorn)                                │
    │                                                                     │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
    │  │Auth      │ │ FL Task  │ │Prediction│ │WebSocket │              │
    │  │Router    │ │ Router   │ │ Router   │ │ Router   │              │
    │  │(JWT)     │ │(CRUD)    │ │(Inference│ │(Pub/Sub) │              │
    │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘              │
    │       │             │            │             │                    │
    │  ┌────┴─────────────┴────────────┴─────────────┴────┐              │
    │  │              Database Layer                       │              │
    │  │  PostgreSQL │ MongoDB │ Redis (Pub/Sub + Cache)   │              │
    │  └──────────────────────┬────────────────────────────┘              │
    └─────────────────────────┼──────────────────────────────────────────┘
                              │
    ┌─────────────────────────┼──────────────────────────────────────────┐
    │                         ▼  FL ENGINE LAYER                         │
    │                  (Flower Framework + PyTorch)                       │
    │                                                                     │
    │  ┌──────────────────────────────────────────────────────┐          │
    │  │              FL Server (Aggregation Hub)              │          │
    │  │  ┌────────────┐ ┌────────────┐ ┌────────────────┐   │          │
    │  │  │ FedAvg     │ │ FedProx    │ │ FedNova        │   │          │
    │  │  │ Strategy   │ │ Strategy   │ │ Strategy       │   │          │
    │  │  └────────────┘ └────────────┘ └────────────────┘   │          │
    │  └───────────────────────┬──────────────────────────────┘          │
    │                          │                                         │
    │      ┌───────────────────┼───────────────────┐                     │
    │      │                   │                   │                     │
    │      ▼                   ▼                   ▼                     │
    │  ┌────────┐         ┌────────┐         ┌────────┐                  │
    │  │Client A│         │Client B│         │Client C│   (Hospital     │
    │  │(NY)    │         │(Chicago│         │(SF)    │    Nodes)        │
    │  └───┬────┘         └───┬────┘         └───┬────┘                  │
    └──────┼──────────────────┼──────────────────┼───────────────────────┘
           │                  │                  │
    ┌──────┼──────────────────┼──────────────────┼───────────────────────┐
    │      ▼  PRIVACY LAYER   ▼                  ▼                       │
    │  ┌──────────────────────────────────────────────────────┐          │
    │  │  Differential Privacy Engine (Renyi DP / DP-SGD)     │          │
    │  │  • Adaptive gradient clipping per-layer              │          │
    │  │  • Calibrated Gaussian noise injection               │          │
    │  │  • Real-time ε budget tracking with auto-halt        │          │
    │  └──────────────────────────────────────────────────────┘          │
    │  ┌──────────────────────────────────────────────────────┐          │
    │  │  Secure Aggregation Engine (Bonawitz SecAgg)          │          │
    │  │  • Pairwise masking via SHA-256 PRG seeds            │          │
    │  │  • Masks cancel during summation (zero-knowledge)    │          │
    │  │  • Dropped-client recovery via seed reconstruction   │          │
    │  └──────────────────────────────────────────────────────┘          │
    └────────────────────────────────────────────────────────────────────┘
           │                  │                  │
    ┌──────┼──────────────────┼──────────────────┼───────────────────────┐
    │      ▼  AI MODULES      ▼                  ▼                       │
    │                                                                     │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
    │  │Module 1  │ │Module 2  │ │Module 3  │ │Module 4  │              │
    │  │EHR       │ │Medical   │ │Drug      │ │Intrusion │              │
    │  │Prediction│ │Imaging   │ │Discovery │ │Detection │              │
    │  │(Sepsis + │ │(Tumor +  │ │(SMILES + │ │(Network  │              │
    │  │ COVID-19)│ │Glaucoma) │ │ Protein) │ │ Traffic) │              │
    │  └──────────┘ └──────────┘ └──────────┘ └──────────┘              │
    └────────────────────────────────────────────────────────────────────┘
```

**In simple words, the system has 5 main layers stacked on top of each other:**

| Layer | What It Does | Technologies |
|-------|-------------|--------------|
| **Frontend** | The visual dashboard you see in the browser | React 18, Vite, TailwindCSS, Recharts |
| **Backend API** | Handles HTTP requests, authentication, task management | FastAPI, Uvicorn, JWT, Pydantic |
| **Database** | Stores tasks, metrics, user accounts, and cached results | PostgreSQL, MongoDB, Redis |
| **FL Engine** | Coordinates training across multiple hospital clients | Flower, PyTorch, FedAvg/FedProx/FedNova |
| **Privacy** | Protects model weights with noise and encryption | Renyi DP, DP-SGD, Bonawitz SecAgg |

---

## 📁 Full Project Structure

Below is the **complete** file listing for the entire FedMedShield project, with every single file explained:

```
d:\FedCare\
│
│── .env                              # Environment variables (ports, DB URLs, DP params)
│── README.md                         # This file
│── ecosystem.config.js               # PM2 process manager config for backend
│── start_demo.ps1                    # 1-click PowerShell boot script
│── setup.sh                          # Linux/Mac auto-installer
│── run_all.py                        # Python orchestration script (alternative launcher)
│── start-frontend.js                 # Node.js script to start Vite dev server
│
├── backend/                          # ══════ FASTAPI BACKEND SERVER ══════
│   ├── __init__.py
│   ├── main.py                       # App entry point: lifespan, CORS, router mounting
│   ├── requirements.txt              # All 30+ Python dependencies pinned
│   │
│   ├── api/                          # Core API endpoints
│   │   ├── auth.py                   # Login/register with bcrypt + JWT tokens
│   │   ├── fl_tasks.py               # CRUD for federated learning tasks
│   │   └── websockets.py             # Real-time WebSocket streaming endpoint
│   │
│   ├── database/                     # Triple-database architecture
│   │   ├── __init__.py
│   │   ├── db_config.py              # Centralized config (reads from .env)
│   │   ├── postgres_db.py            # PostgreSQL async connection + table init
│   │   ├── postgres_models.py        # SQLAlchemy ORM models (Users, Tasks)
│   │   ├── mongo_db.py               # MongoDB async client (motor)
│   │   ├── mongo_models.py           # Document schemas for FL metrics
│   │   ├── redis_db.py               # Redis connection manager
│   │   └── redis_cache.py            # Redis pub/sub + caching layer
│   │
│   ├── middleware/                    # Request pipeline interceptors
│   │   ├── __init__.py
│   │   ├── auth_middleware.py         # JWT token verification middleware
│   │   └── cors_middleware.py         # CORS headers for browser access
│   │
│   ├── models/                       # Pydantic request/response schemas
│   │   ├── __init__.py               # Re-exports all model classes
│   │   ├── auth_models.py            # LoginRequest, RegisterRequest, TokenResponse
│   │   ├── fl_models.py              # TaskCreate, TaskStatus, RoundMetrics
│   │   ├── prediction_models.py      # PatientData, PredictionResult
│   │   ├── imaging_models.py         # ImageUpload, DiagnosisResult
│   │   ├── drug_models.py            # DrugScreen, AffinityResult
│   │   └── ids_models.py             # NetworkPacket, ThreatDetection
│   │
│   ├── routers/                      # Modular FastAPI routers (one per domain)
│   │   ├── __init__.py
│   │   ├── auth_router.py            # POST /api/auth/login, /api/auth/register
│   │   ├── fl_router.py              # POST /api/tasks, GET /api/tasks/{id}
│   │   ├── prediction_router.py      # POST /api/predict/ehr
│   │   ├── imaging_router.py         # POST /api/imaging/diagnose
│   │   ├── drug_router.py            # POST /api/drug/screen
│   │   └── ids_router.py             # POST /api/ids/analyze
│   │
│   └── utils/                        # Shared utilities
│       ├── __init__.py
│       └── jwt_utils.py              # JWT encode/decode with HS256
│
├── clients/                          # ══════ HOSPITAL CLIENT NODES ══════
│   ├── __init__.py
│   ├── hospital_client.py            # Flower client wrapper (connects to FL server)
│   └── client_config.json            # JSON config for 4 hospital nodes (ports, names)
│
├── data/                             # ══════ DATA GENERATORS & LOADERS ══════
│   ├── __init__.py
│   │
│   ├── ehr_data/                     # Electronic Health Record generator
│   │   ├── __init__.py
│   │   └── data_loader.py            # Generates synthetic patient vitals, labs, outcomes
│   │
│   ├── imaging_data/                 # Medical image simulator
│   │   ├── __init__.py
│   │   └── data_loader.py            # Creates synthetic X-ray/CT tensors + labels
│   │
│   ├── drug_data/                    # Drug-protein binding generator
│   │   ├── __init__.py
│   │   └── data_loader.py            # Generates SMILES strings + protein sequences
│   │
│   └── network_data/                 # Network traffic generator (for IDS)
│       ├── __init__.py
│       └── data_loader.py            # Creates synthetic network flow features
│
├── evaluation/                       # ══════ IEEE PUBLICATION PLOTTER ══════
│   ├── __init__.py
│   ├── global_eval.py                # Loads checkpoint + runs global evaluation
│   ├── metrics.py                    # Accuracy, F1, AUC-ROC, Precision, Recall
│   ├── plotter.py                    # Matplotlib → PDF (300 DPI, Times New Roman)
│   ├── compare_federated_vs_central.py  # Head-to-head comparison benchmark
│   └── results/
│       └── benchmark_comparison.json # Pre-computed comparison metrics
│
├── fl_engine/                        # ══════ FEDERATED LEARNING CORE ══════
│   ├── __init__.py
│   ├── server.py                     # FL Server: round management, client selection
│   ├── client.py                     # FL Client: local training loop
│   ├── strategy.py                   # FedAvg, FedProx, FedNova implementations
│   ├── aggregator.py                 # Weight aggregation with DP noise injection
│   └── orchestrator.py               # Spawns server + N clients as subprocesses
│
├── frontend/                         # ══════ REACT DASHBOARD ══════
│   ├── package.json                  # NPM dependencies (React, Recharts, Zustand)
│   ├── tsconfig.json                 # TypeScript compiler configuration
│   ├── vite.config.ts                # Vite bundler configuration
│   ├── tailwind.config.ts            # TailwindCSS theme customization
│   ├── postcss.config.js             # PostCSS plugins (autoprefixer, tailwind)
│   ├── index.html                    # Root HTML entry point
│   │
│   └── src/
│       ├── main.tsx                  # React DOM root mount
│       ├── App.tsx                   # Router setup (BrowserRouter + Routes)
│       ├── index.css                 # Global CSS (Tailwind imports + custom styles)
│       │
│       ├── types/                    # TypeScript type definitions
│       │   ├── hospital.types.ts     # HospitalNode, HospitalStatus, ConnectionInfo
│       │   ├── fl.types.ts           # FLTask, FLRound, FLConfig, AggregationResult
│       │   ├── prediction.types.ts   # PatientData, PredictionResult, RiskLevel
│       │   ├── imaging.types.ts      # ImageScan, DiagnosisResult, TumorType
│       │   ├── drug.types.ts         # DrugCandidate, ProteinTarget, AffinityScore
│       │   └── ids.types.ts          # NetworkPacket, ThreatLevel, AttackType
│       │
│       ├── components/               # Reusable UI components
│       │   ├── Header.tsx            # Top bar with user info + notifications
│       │   ├── Sidebar.tsx           # Side navigation with route links
│       │   │
│       │   ├── cards/                # Data display cards
│       │   │   ├── StatCard.tsx       # Animated metric card with icon
│       │   │   ├── MetricCard.tsx     # Compact inline metric display
│       │   │   ├── AlertCard.tsx      # Warning/error notification card
│       │   │   ├── HospitalStatusCard.tsx  # Node connection status card
│       │   │   └── PredictionResultCard.tsx # AI prediction result display
│       │   │
│       │   ├── charts/               # Recharts-based visualizations
│       │   │   ├── LossChart.tsx      # Real-time training loss line chart
│       │   │   ├── AccuracyChart.tsx   # Accuracy convergence area chart
│       │   │   ├── FLRoundChart.tsx    # Per-round bar chart
│       │   │   ├── HospitalRadarChart.tsx  # Multi-axis radar comparison
│       │   │   └── PrivacyBudgetGauge.tsx  # ε budget consumption gauge
│       │   │
│       │   ├── forms/                # Data input forms
│       │   │   ├── PatientDataForm.tsx # EHR prediction input form
│       │   │   ├── ImageUploadForm.tsx # Medical image upload form
│       │   │   ├── DrugScreenForm.tsx  # Drug SMILES input form
│       │   │   └── TaskForm.tsx       # FL task creation wizard
│       │   │
│       │   └── layout/               # Layout primitives
│       │       ├── Layout.tsx         # Page wrapper with sidebar + header
│       │       ├── LoadingSpinner.tsx  # Animated loading indicator
│       │       ├── Navbar.tsx         # Alternative navigation bar
│       │       └── Sidebar.tsx        # Extended sidebar variant
│       │
│       ├── pages/                    # Route-level page components
│       │   ├── Dashboard.tsx          # Main overview (live simulation engine)
│       │   ├── NetworkMap.tsx         # FL network topology visualization
│       │   ├── Privacy.tsx            # DP + SecAgg configuration panel
│       │   ├── PrivacyReport.tsx      # Detailed privacy budget report
│       │   ├── Nodes.tsx              # Hospital node management
│       │   ├── HospitalNodes.tsx      # Extended node monitoring
│       │   ├── Settings.tsx           # System configuration
│       │   ├── Login.tsx              # JWT login page
│       │   ├── DiseasePrediction.tsx   # EHR prediction page (Module 1)
│       │   ├── MedicalImaging.tsx     # Image diagnosis page (Module 2)
│       │   ├── DrugDiscovery.tsx      # Drug screening page (Module 3)
│       │   └── IntrusionDetection.tsx # IDS monitoring page (Module 4)
│       │
│       ├── context/                  # React Context providers
│       │   ├── AuthContext.tsx        # Authentication state (JWT token, user)
│       │   └── FLContext.tsx          # Federated Learning state (tasks, rounds)
│       │
│       ├── hooks/                    # Custom React hooks
│       │   ├── useAuth.ts            # Login/logout/register hook
│       │   ├── useFLStatus.ts        # FL task status polling hook
│       │   ├── useHospitalNodes.ts   # Node health monitoring hook
│       │   ├── usePrediction.ts      # Prediction API call hook
│       │   ├── useWebSocket.ts       # Single WebSocket connection hook
│       │   └── useWebSockets.ts      # Multi-channel WebSocket hook
│       │
│       ├── services/                 # API service layer (Axios)
│       │   ├── api.ts                # Base Axios instance configuration
│       │   ├── api.service.ts        # Generic CRUD service
│       │   ├── fl.service.ts         # FL task CRUD operations
│       │   ├── prediction.service.ts # EHR prediction API calls
│       │   ├── imaging.service.ts    # Imaging diagnosis API calls
│       │   ├── drug.service.ts       # Drug screening API calls
│       │   ├── ids.service.ts        # IDS analysis API calls
│       │   └── socket.service.ts     # WebSocket connection manager
│       │
│       ├── store/                    # Zustand global state
│       │   └── useStore.ts           # Central state store (tasks, metrics)
│       │
│       └── utils/                    # Shared utilities
│           ├── constants.ts          # App-wide constants (colors, labels)
│           ├── formatters.ts         # Number/date formatting helpers
│           └── validators.ts         # Form input validation functions
│
├── modules/                          # ══════ 4 AI NEURAL NETWORK ARCHITECTURES ══════
│   ├── __init__.py
│   │
│   ├── module1_ehr/                  # Electronic Health Records
│   │   ├── __init__.py
│   │   ├── ehr_model.py              # Multi-head residual network (PyTorch)
│   │   ├── covid_model.py            # COVID-19 mortality sub-model
│   │   ├── sepsis_model.py           # Sepsis urgency sub-model
│   │   └── train.py                  # Local training loop with DP-SGD
│   │
│   ├── module2_imaging/              # Medical Imaging
│   │   ├── __init__.py
│   │   ├── imaging_model.py          # ResNet-18 with transfer learning
│   │   ├── tumor_detector.py         # Tumor classification head
│   │   ├── glaucoma_detector.py      # Glaucoma detection head
│   │   └── train.py                  # Image training with augmentation
│   │
│   ├── module3_drug/                 # Drug Discovery
│   │   ├── __init__.py
│   │   ├── drug_model.py             # Dual-encoder binding affinity predictor
│   │   ├── compound_encoder.py       # SMILES string encoder (character-level CNN)
│   │   └── train.py                  # Drug-protein binding training loop
│   │
│   └── module4_ids/                  # Intrusion Detection System
│       ├── __init__.py
│       ├── ids_model.py              # Multi-scale network traffic analyzer
│       └── train.py                  # IDS training with Focal Loss
│
├── privacy/                          # ══════ PRIVACY PROTECTION ENGINES ══════
│   ├── __init__.py                   # Exports DP + SecAgg engines
│   ├── differential_privacy.py       # Renyi DP engine with adaptive clipping
│   ├── noise_utils.py                # Gaussian noise calibration utilities
│   └── secure_aggregation.py         # Bonawitz pairwise masking protocol
│
└── logs/                             # Runtime log files (auto-created)
    ├── backend-out.log               # Backend stdout
    └── backend-error.log             # Backend stderr
```

**Total file count:** 100+ files across 30+ directories.

---

## 🔧 Phase 1 — Project Setup

This phase creates the entire project skeleton and configures all tools.

### Step 1.1 — Folder Structure

Every folder listed in the directory tree above was created to enforce separation of concerns:
- `backend/` holds only server-side Python code
- `frontend/` holds only client-side TypeScript/React code
- `modules/` holds only PyTorch model architectures
- `privacy/` holds only privacy-preserving algorithms
- `fl_engine/` holds only the federated learning coordination logic
- `data/` holds only data generators and loaders

### Step 1.2 — Environment Variables (`.env`)

The `.env` file centralizes every configurable value so nothing is hardcoded:

| Variable | Purpose | Default |
|----------|---------|---------|
| `BACKEND_HOST` | API server bind address | `0.0.0.0` |
| `BACKEND_PORT` | API server port | `8000` |
| `SECRET_KEY` | JWT signing secret | `fedmedshield_secret_key_...` |
| `JWT_ALGORITHM` | Token algorithm | `HS256` |
| `FL_TOTAL_ROUNDS` | Number of federated rounds | `20` |
| `FL_LOCAL_EPOCHS` | Local training epochs per round | `5` |
| `FL_MIN_CLIENTS` | Minimum clients required | `3` |
| `FL_FRACTION_FIT` | Fraction of clients per round | `0.8` |
| `DP_EPSILON` | Initial privacy budget | `1.0` |
| `DP_DELTA` | DP failure probability | `1e-5` |
| `DP_MAX_BUDGET` | Auto-halt threshold | `10.0` |
| `MONGODB_URL` | MongoDB connection string | `mongodb://localhost:27017/` |
| `POSTGRES_URL` | PostgreSQL connection string | `postgresql://admin:password@localhost:5432/fedmedshield` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |

### Step 1.3 — Backend Dependencies (`requirements.txt`)

All 30+ Python packages are version-pinned for reproducibility:

| Category | Packages |
|----------|----------|
| **Web Framework** | FastAPI 0.115, Uvicorn 0.30, WebSockets 12 |
| **Validation** | Pydantic 2.9, Pydantic-Settings 2.5 |
| **Security** | python-jose 3.3, passlib 1.7, cryptography 43 |
| **Databases** | PyMongo 4.9, SQLAlchemy 2.0, asyncpg 0.29, Redis 5.1 |
| **Deep Learning** | PyTorch 2.4, TorchVision 0.19, TorchAudio 2.4 |
| **Data Science** | NumPy 1.26, Pandas 2.2, Scikit-learn 1.5 |
| **Image Processing** | OpenCV 4.10, Pillow 10.4 |
| **Federated Learning** | Flower (flwr) 1.11 |
| **Utilities** | tqdm, python-dotenv, httpx, aiofiles |

### Step 1.4 — Frontend Dependencies (`package.json`)

| Package | Version | Purpose |
|---------|---------|---------|
| `react` | 18.3 | UI component library |
| `react-dom` | 18.3 | DOM rendering engine |
| `react-router-dom` | 6.24 | Client-side routing |
| `recharts` | 2.12 | Chart visualizations |
| `zustand` | 4.5 | Lightweight global state management |
| `axios` | 1.7 | HTTP client for API calls |
| `lucide-react` | 0.395 | Icon library |
| `clsx` | 2.1 | Conditional class name utility |
| `tailwind-merge` | 2.3 | TailwindCSS class conflict resolver |

### Step 1.5 — Build Configuration

- **`vite.config.ts`** — Vite build tool with React plugin, dev server proxy to backend
- **`tsconfig.json`** — Strict TypeScript compiler settings
- **`tailwind.config.ts`** — Custom theme with glassmorphism colors
- **`postcss.config.js`** — PostCSS pipeline with TailwindCSS + Autoprefixer

---

## 📐 Phase 2 — Type Definitions (Frontend)

TypeScript type definitions create a strict contract between the frontend and backend. If the backend sends data in a different shape than expected, TypeScript catches the error at compile time instead of crashing at runtime.

### `hospital.types.ts`
Defines the shape of a hospital node in the federated network:
- `HospitalNode` — id, name, location, status (online/offline/training), port
- `HospitalStatus` — connection health, last ping, data size, current round
- `ConnectionInfo` — WebSocket connection state, latency, throughput

### `fl.types.ts`
Defines federated learning task structures:
- `FLTask` — task ID, type (ehr/imaging/drug/ids), status, created timestamp
- `FLRound` — round number, participating clients, loss, accuracy, duration
- `FLConfig` — total rounds, local epochs, min clients, fraction fit, strategy
- `AggregationResult` — aggregated weights hash, global loss, global accuracy

### `prediction.types.ts`
Defines EHR prediction input/output:
- `PatientData` — age, vitals, lab values, comorbidities, medications
- `PredictionResult` — disease probability, risk level, confidence score
- `RiskLevel` — enum: LOW, MODERATE, HIGH, CRITICAL

### `imaging.types.ts`
Defines medical imaging pipeline:
- `ImageScan` — base64 image data, modality (X-ray/CT/MRI), body region
- `DiagnosisResult` — classification, confidence, heatmap coordinates
- `TumorType` — enum: BENIGN, MALIGNANT, UNCERTAIN

### `drug.types.ts`
Defines drug discovery interface:
- `DrugCandidate` — SMILES string, molecular weight, properties
- `ProteinTarget` — amino acid sequence, binding site, 3D coordinates
- `AffinityScore` — predicted binding affinity (pIC50), confidence

### `ids.types.ts`
Defines intrusion detection interface:
- `NetworkPacket` — source/dest IP, port, protocol, payload size, flags
- `ThreatLevel` — enum: NORMAL, SUSPICIOUS, MALICIOUS, CRITICAL
- `AttackType` — enum: DoS, DDoS, MITM, SQLInjection, XSS, PortScan

---

## 🧬 Phase 3 — Synthetic Data Generation

Real patient data cannot be shared or used in demonstrations. FedMedShield generates **high-fidelity synthetic data** that statistically resembles real clinical data.

### Key Technique: Dirichlet Distribution for Non-IID Data

In the real world, different hospitals see different types of patients. A children's hospital has more pediatric cases, a trauma center has more emergency cases. This uneven distribution is called **Non-IID** (Non-Independent and Identically Distributed).

We simulate this using the **Dirichlet distribution** with concentration parameter α:
- **α = 0.1** → Extremely uneven (each hospital specializes in one disease)
- **α = 1.0** → Moderately uneven (realistic hospital variation)
- **α = 100** → Nearly uniform (all hospitals have similar data)

### Data Generators

| Generator | File | Records | Features |
|-----------|------|---------|----------|
| **EHR** | `data/ehr_data/data_loader.py` | 10,000 patients | 45 vitals + labs + medications |
| **Imaging** | `data/imaging_data/data_loader.py` | 5,000 scans | 224×224 pixel tensors + labels |
| **Drug** | `data/drug_data/data_loader.py` | 8,000 pairs | SMILES strings + protein sequences |
| **Network** | `data/network_data/data_loader.py` | 50,000 packets | 78 traffic features + attack labels |

Each generator:
1. Creates realistic feature distributions using NumPy random generators
2. Introduces correlations between features (e.g., high fever correlates with elevated WBC)
3. Splits data across N hospital clients using Dirichlet partitioning
4. Returns PyTorch DataLoaders ready for training

---

## 🧠 Phase 4 — AI Model Architectures (PyTorch)

FedMedShield implements **4 distinct neural network architectures**, each targeting a different healthcare AI task.

### Module 1: EHR Prediction (`modules/module1_ehr/`)

**Purpose:** Predict disease risk from Electronic Health Records (patient vitals, lab results, medications).

**Architecture — Multi-Head Residual Network:**
```
Input (45 features)
    │
    ▼
┌─────────────────┐
│  Shared Trunk    │  ← 3 residual blocks with batch norm + dropout
│  (Feature        │     Each block: Linear → BatchNorm → ReLU → Dropout → Linear → Skip
│   Extraction)    │
└────────┬────────┘
         │
    ┌────┼────┐
    ▼    ▼    ▼
┌──────┐┌──────┐┌──────┐
│Head 1││Head 2││Head 3│  ← 3 independent classification heads
│Sepsis││COVID ││Risk  │     Each: Linear → ReLU → Linear → Sigmoid
└──────┘└──────┘└──────┘
```

- **Head 1 (Sepsis):** Binary classification — will this patient develop sepsis within 6 hours?
- **Head 2 (COVID-19):** Mortality prediction — what is the probability of COVID-19 death?
- **Head 3 (Risk):** Multi-class risk stratification — LOW / MODERATE / HIGH / CRITICAL

**Key files:**
- `ehr_model.py` — The main multi-head residual network (13,762 bytes)
- `sepsis_model.py` — Sepsis-specific sub-model with time-series attention (7,844 bytes)
- `covid_model.py` — COVID-19 mortality predictor with clinical feature weighting (6,203 bytes)
- `train.py` — Training loop with DP-SGD integration (12,583 bytes)

### Module 2: Medical Imaging (`modules/module2_imaging/`)

**Purpose:** Detect tumors and glaucoma from medical images (X-rays, CT scans, retinal images).

**Architecture — ResNet-18 with Transfer Learning:**
```
Input Image (3 × 224 × 224)
    │
    ▼
┌──────────────────────┐
│  ResNet-18 Backbone   │  ← Pre-trained on ImageNet (frozen early layers)
│  (Conv1 → Layer4)     │     Fine-tuned later layers for medical images
└──────────┬───────────┘
           │
      ┌────┼────┐
      ▼         ▼
┌──────────┐ ┌──────────┐
│ Tumor    │ │ Glaucoma │  ← Two detection heads
│ Detector │ │ Detector │     Tumor: 3-class (benign/malignant/uncertain)
└──────────┘ └──────────┘     Glaucoma: binary (positive/negative)
```

- Uses transfer learning to leverage ImageNet weights
- Data augmentation: random rotation, horizontal flip, color jitter, random crop
- Gradient-weighted Class Activation Mapping (Grad-CAM) for explainability

**Key files:**
- `imaging_model.py` — ResNet-18 backbone with dual heads (11,732 bytes)
- `tumor_detector.py` — Tumor classification head (606 bytes)
- `glaucoma_detector.py` — Glaucoma detection head (645 bytes)
- `train.py` — Training with augmentation pipeline (9,452 bytes)

### Module 3: Drug Discovery (`modules/module3_drug/`)

**Purpose:** Predict how strongly a drug molecule will bind to a target protein (binding affinity prediction).

**Architecture — Dual-Encoder with Cross-Attention:**
```
Drug SMILES String          Protein Sequence
    │                           │
    ▼                           ▼
┌──────────────┐        ┌──────────────┐
│ Compound     │        │ Protein      │
│ Encoder      │        │ Encoder      │
│ (Char-CNN)   │        │ (1D-CNN)     │
└──────┬───────┘        └──────┬───────┘
       │                       │
       └───────────┬───────────┘
                   │
              ┌────▼────┐
              │  Cross   │
              │Attention │
              │  Fusion  │
              └────┬────┘
                   │
              ┌────▼────┐
              │ Affinity │  ← Predicts pIC50 (binding strength)
              │Predictor │     Higher = stronger binding
              └─────────┘
```

- **Compound Encoder:** Converts SMILES strings (e.g., `CC(=O)Oc1ccccc1C(=O)O` for Aspirin) into fixed-length embeddings using character-level CNN
- **Protein Encoder:** Converts amino acid sequences into embeddings using 1D convolutions
- **Cross-Attention Fusion:** Learns which parts of the drug interact with which parts of the protein
- **Output:** Predicted pIC50 value (log-scale binding affinity)

**Key files:**
- `drug_model.py` — Full dual-encoder architecture (13,429 bytes)
- `compound_encoder.py` — SMILES character-level CNN encoder (8,167 bytes)
- `train.py` — Training loop with MSE loss on pIC50 (9,244 bytes)

### Module 4: Intrusion Detection (`modules/module4_ids/`)

**Purpose:** Detect cyber attacks on the FL network itself by analyzing network traffic patterns.

**Architecture — Multi-Scale Network Traffic Analyzer:**
```
Network Flow Features (78 dims)
    │
    ▼
┌─────────────────────────────────────────┐
│         Multi-Scale Feature Extractor    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │ Scale 1  │ │ Scale 2  │ │ Scale 3  ││
│  │ (Fine)   │ │ (Medium) │ │ (Coarse) ││
│  │ kernel=3 │ │ kernel=5 │ │ kernel=7 ││
│  └────┬─────┘ └────┬─────┘ └────┬─────┘│
│       └──────────┬──┴────────────┘      │
│                  │ Concatenate           │
└──────────────────┼──────────────────────┘
                   │
              ┌────▼────┐
              │ Focal   │  ← Handles class imbalance
              │ Loss    │     (99% normal, 1% attacks)
              │Classifier│
              └─────────┘
```

- **Multi-scale analysis:** Captures both fine-grained (single-packet) and coarse-grained (flow-level) attack signatures
- **Focal Loss:** Addresses extreme class imbalance where attacks are rare events
- **Attack types detected:** DoS, DDoS, Man-in-the-Middle, SQL Injection, XSS, Port Scanning

**Key files:**
- `ids_model.py` — Multi-scale analyzer with focal loss (13,645 bytes)
- `train.py` — Training with class-weighted sampling (11,751 bytes)

---

## 🔒 Phase 5 — Privacy Engines

FedMedShield integrates reviewed, formal privacy and cryptographic mechanisms designed for distributed healthcare federated learning.

### 5.1 Validated Differential Privacy Engine (`privacy/differential_privacy.py`)

Model weights and gradient updates in federated learning remain vulnerable to reconstruction, membership inference, and attribute extraction if shared unprotected. The differential privacy engine provides formal privacy guarantees:

1. **Validated Rényi Differential Privacy (RDP):**
   - Employs the reviewed `opacus.accountants.RDPAccountant` from PyTorch Opacus for rigorous accounting under subsampled Gaussian mechanisms.
   - Computes explicit $(\varepsilon, \delta)$ bounds across arbitrary compositions.

2. **Explicit Privacy Scoping:**
   - **Client-Level DP (`client_level`):** Calibrates noise against cohort sampling rate $q = \frac{K}{N}$ across federation rounds.
   - **Patient-Level DP (`patient_level`):** Calibrates noise against local minibatch sampling rate $q = \frac{B}{D}$.

3. **Pre-Release Privacy Budget Gating:**
   - Evaluates remaining privacy budget before releasing any local model updates.
   - If projected cumulative $\varepsilon$ reaches `target_epsilon`, the client withholds the trained weights and returns the unmodified global model weights, emitting a `DP_BUDGET_EXHAUSTED` security audit event.

4. **Metadata Bypass Prevention:**
   - Suppresses raw gradient norms, adaptive clipping thresholds, and local dataset sizes from server-visible telemetry to prevent side-channel information leakage.

| Parameter | Symbol | Default | Meaning |
|-----------|--------|---------|---------|
| Target Epsilon | ε_target | 6.0 | Hard privacy budget limit (halts updates if exceeded) |
| Delta | δ | 1e-5 | Privacy failure probability |
| Noise Multiplier | σ | 1.0 | Standard deviation of Gaussian perturbation relative to clip bound |
| Initial Clip Bound | C | 1.0 | Adaptive L2-norm gradient clipping threshold |

### 5.2 Security Audit Logger (`privacy/audit_logger.py`)

Maintains an immutable, append-only JSON Lines security audit ledger (`logs/security_audit.jsonl`) recording all critical security and privacy events:
- `KEY_ROTATION`: Participant X25519 ephemeral key pair rotation per round.
- `DP_INITIALIZED` / `DP_STEP_APPLIED`: Accounting steps with recorded $(\varepsilon, \delta)$ spend.
- `DP_BUDGET_EXHAUSTED`: Automatic client update suppression when budget is depleted.
- `SECOBJ_AGGREGATED`: Server-side completion of secure aggregation.

### 5.3 Reviewed Secure Aggregation Engine (`privacy/secure_aggregation.py`)

Implements reviewed cryptographic primitives derived from upstream Flower SecAgg+ (`flwr.common.secure_aggregation`):

1. **X25519 ECDH Key Agreement & Forward Secrecy:**
   - Uses `cryptography.hazmat.primitives.asymmetric.x25519` for Elliptic-Curve Diffie-Hellman shared secret establishment.
   - Ephemeral key rotation per round ensures forward secrecy (`rotate_keys()`).

2. **Shamir Secret Sharing for Dropout Tolerance ($t$-out-of-$U$):**
   - Splits private masks and seeds into $U$ shares using Flower's Shamir implementation (`flwr.common.secure_aggregation.crypto.shamir`).
   - If up to $U - t$ hospital clients drop out mid-round, surviving nodes reconstruct surviving masks without exposing offline private weights.

3. **Stochastic Quantization:**
   - Quantizes continuous float weights into discrete integers with stochastic rounding (`flwr.common.secure_aggregation.quantization`).
   - Accounts for zero-offset accumulation across $K$ participants to guarantee exact algebraic cancellation.

```
Hospital Node A ──► X25519 ECDH ──► Shamir Shares (t-out-of-U) ──► Quantized Masked Weights ┐
Hospital Node B ──► X25519 ECDH ──► Shamir Shares (t-out-of-U) ──► Quantized Masked Weights ┼─► Flower SecAvg Aggregator
Hospital Node C ──► X25519 ECDH ──► Shamir Shares (t-out-of-U) ──► Quantized Masked Weights ┘   (Symmetric masks cancel)
```

> [!IMPORTANT]
> **Threat Model & Cryptographic Assumptions:**
> - **Honest-but-curious Server:** The aggregator honestly follows the protocol but attempts to infer individual weights.
> - **Non-Collusion Bound:** Cryptographic privacy holds provided fewer than the threshold $t$ clients collude with the server or each other.
> - **Necessity of Differential Privacy:** Secure Aggregation protects updates *in-flight* to the server; it does **not** protect against reconstruction attacks on the final aggregated model. Hence, DP perturbation applied prior to aggregation is mandatory for end-to-end clinical data confidentiality.

---

## ⚙️ Phase 6 — Federated Learning Engine

The FL Engine coordinates the entire distributed training process.

### 6.1 FL Server (`fl_engine/server.py`)

The central orchestrator that manages the training lifecycle:

1. **Client Registration:** Hospitals connect to the server and register their capabilities (GPU, dataset size)
2. **Client Selection:** Each round, the server selects a fraction (`FL_FRACTION_FIT = 0.8`) of available clients
3. **Weight Distribution:** Server sends current global model weights to selected clients
4. **Aggregation:** Collects trained weights from clients and aggregates using the chosen strategy
5. **Evaluation:** Runs global evaluation on a held-out test set after each round
6. **Checkpointing:** Saves model weights to disk after each round for recovery

### 6.2 FL Client (`fl_engine/client.py`)

Each hospital runs a client that:
1. Receives global weights from the server
2. Loads its local (private) dataset
3. Trains the model locally for `FL_LOCAL_EPOCHS` epochs
4. Applies Differential Privacy noise to the trained weights
5. Applies Secure Aggregation masks
6. Sends the masked, noisy weights back to the server
7. Reports local metrics (loss, accuracy, dataset size)

### 6.3 Aggregation Strategies (`fl_engine/strategy.py`)

Three strategies are implemented, each with different tradeoffs:

| Strategy | Paper | How It Works | Best For |
|----------|-------|-------------|----------|
| **FedAvg** | McMahan 2017 | Weighted average by dataset size | IID data, stable networks |
| **FedProx** | Li et al. 2020 | FedAvg + proximal term to penalize drift from global model | Non-IID data, heterogeneous clients |
| **FedNova** | Wang et al. 2020 | Normalized averaging that accounts for varying local steps | Clients with different compute power |

**FedAvg formula:**
```
w_global = Σ (n_k / n_total) × w_k

Where:
  w_k       = weights from hospital k
  n_k       = number of samples at hospital k
  n_total   = total samples across all hospitals
```

**FedProx addition:**
```
Loss_k = Original_Loss + (μ/2) × ||w_k - w_global||²

The proximal term μ prevents hospital models from drifting too far
from the global model, which helps with Non-IID data.
```

### 6.4 Weight Aggregator (`fl_engine/aggregator.py`)

Handles the actual mathematical aggregation:
1. Receives a list of client weight dictionaries
2. Applies the selected strategy (FedAvg/FedProx/FedNova)
3. Injects DP noise into the aggregated result
4. Validates numerical stability (no NaN/Inf values)
5. Returns the new global model state dict

### 6.5 Orchestrator (`fl_engine/orchestrator.py`)

A process manager that spawns the FL server and multiple FL clients as Python subprocesses:
- Reads configuration from `.env`
- Starts the FL server on `FL_SERVER_PORT`
- Spawns N client processes, each configured for a different hospital
- Monitors process health and restarts crashed clients
- Streams metrics to the backend via Redis pub/sub

---

## 🌐 Phase 7 — Backend API (FastAPI)

The backend is the bridge between the frontend dashboard and the FL engine.

### 7.1 Application Entry Point (`backend/main.py`)

- Uses FastAPI's `lifespan` context manager for graceful startup/shutdown
- Initializes all three databases (PostgreSQL, MongoDB, Redis) with fallback resilience
- Mounts 6 domain-specific routers under the `/api` prefix
- Configures CORS to allow browser access from any origin
- Exposes `/health` endpoint for monitoring

### 7.2 Triple-Database Architecture

We use three different databases because each excels at a different task:

| Database | Purpose | Why This One |
|----------|---------|-------------|
| **PostgreSQL** | User accounts, FL tasks, structured metadata | ACID transactions, relational integrity |
| **MongoDB** | FL round metrics, training logs, nested documents | Schema flexibility, nested JSON support |
| **Redis** | Real-time pub/sub, metric caching, WebSocket relay | Sub-millisecond latency, pub/sub channels |

**PostgreSQL Models (`postgres_models.py`):**
- `User` — username, hashed password, role, created_at
- `FLTask` — task_id, type, status, rounds_completed, started_at, finished_at

**MongoDB Documents (`mongo_models.py`):**
- `RoundMetric` — task_id, round_number, loss, accuracy, participating_clients, timestamp
- `PrivacyLog` — epsilon_consumed, delta, noise_applied, budget_remaining

**Redis Channels (`redis_cache.py`):**
- `fl:metrics:{task_id}` — Real-time training metrics stream
- `fl:status:{task_id}` — Task status updates
- `fl:budget:{task_id}` — Privacy budget consumption updates

### 7.3 Authentication System

- **Registration:** Passwords hashed with bcrypt (10 salt rounds) before storage
- **Login:** Returns a JWT token with 24-hour expiration
- **Protected routes:** JWT middleware verifies token on every request to `/api/tasks`, `/api/predict`, etc.
- **Token format:** Header + Payload (user_id, role, exp) + HS256 Signature

### 7.4 API Routers

| Router | Prefix | Endpoints |
|--------|--------|-----------|
| `auth_router.py` | `/api/auth` | `POST /login`, `POST /register` |
| `fl_router.py` | `/api/tasks` | `POST /`, `GET /`, `GET /{id}`, `POST /{id}/start`, `DELETE /{id}` |
| `prediction_router.py` | `/api/predict` | `POST /ehr` |
| `imaging_router.py` | `/api/imaging` | `POST /diagnose` |
| `drug_router.py` | `/api/drug` | `POST /screen` |
| `ids_router.py` | `/api/ids` | `POST /analyze` |

### 7.5 WebSocket Streaming (`api/websockets.py`)

Real-time bidirectional communication for live dashboard updates:
- Client connects to `ws://localhost:8000/ws/metrics/{task_id}`
- Server subscribes to the Redis pub/sub channel for that task
- As each FL round completes, metrics are published to Redis
- WebSocket relay pushes JSON metrics to the browser instantly
- No polling needed — true server-push architecture

---

## 🖥 Phase 8 — Frontend Dashboard (React + Vite)

The dashboard provides a professional, real-time visualization of the federated learning system.

### 8.1 Application Shell

**`App.tsx`** — Sets up React Router with the following routes:

| Route | Page Component | Description |
|-------|---------------|-------------|
| `/login` | `Login.tsx` | JWT authentication page |
| `/dashboard` | `Dashboard.tsx` | Main overview with live simulation |
| `/network` | `NetworkMap.tsx` | FL network topology visualization |
| `/privacy` | `Privacy.tsx` | DP + SecAgg configuration |
| `/privacy-report` | `PrivacyReport.tsx` | Detailed privacy audit |
| `/nodes` | `Nodes.tsx` | Hospital node management |
| `/hospital-nodes` | `HospitalNodes.tsx` | Extended node monitoring |
| `/settings` | `Settings.tsx` | System configuration |
| `/disease-prediction` | `DiseasePrediction.tsx` | Module 1 UI |
| `/medical-imaging` | `MedicalImaging.tsx` | Module 2 UI |
| `/drug-discovery` | `DrugDiscovery.tsx` | Module 3 UI |
| `/intrusion-detection` | `IntrusionDetection.tsx` | Module 4 UI |

### 8.2 Dashboard Page (`Dashboard.tsx`)

The centerpiece of the UI. Features a **built-in simulation engine** that generates realistic FL training data in real-time, even without the backend running:

- **4 animated stat cards:** Active Nodes, Current Round, Privacy Budget (ε), Global Accuracy
- **Hospital node badges:** Visual indicators for NY, Chicago, SF, Austin nodes
- **Live loss convergence chart:** Plots training loss with realistic exponential decay
- **Live accuracy chart:** Shows accuracy climbing with realistic noise
- **Round log table:** Scrolling table with per-round metrics and pulsing "LIVE" badge
- **Status banner:** Shows lifecycle stages (Initializing → Training → Aggregation → Completed)
- **Start buttons:** "Start EHR Task" and "Start Imaging Task" launch the simulation

The simulation engine uses mathematical models to generate realistic curves:
```
loss(round) = initial_loss × exp(-decay_rate × round) + noise
accuracy(round) = max_accuracy × (1 - exp(-growth_rate × round)) + noise
```

### 8.3 Network Map (`NetworkMap.tsx`)

Visual topology of the federated network:
- Central server node in the middle
- Hospital client nodes around the perimeter
- Animated connection lines showing data flow direction
- Color-coded status: green (online), yellow (training), red (offline)
- Real-time latency display on each connection

### 8.4 Module-Specific Pages

Each of the 4 AI modules has a dedicated page:

**Disease Prediction (`DiseasePrediction.tsx`):**
- Patient data input form (vitals, labs, medications)
- Risk level visualization with color-coded severity
- Confidence score meter

**Medical Imaging (`MedicalImaging.tsx`):**
- Drag-and-drop image upload area
- Before/after comparison view
- Detection results with bounding boxes
- Confidence percentage bar

**Drug Discovery (`DrugDiscovery.tsx`):**
- SMILES string input with structure preview
- Protein target selector
- Binding affinity score gauge
- Molecular property table

**Intrusion Detection (`IntrusionDetection.tsx`):**
- Real-time network traffic stream
- Threat level indicator with alert system
- Attack type classification breakdown
- Historical threat timeline chart

### 8.5 Component Library

**Cards (5 components):**
- `StatCard` — Animated metric with icon, gradient background
- `MetricCard` — Compact inline metric display
- `AlertCard` — Warning/error notification with dismiss
- `HospitalStatusCard` — Node connection status with ping
- `PredictionResultCard` — AI prediction result with confidence

**Charts (5 components):**
- `LossChart` — Recharts line chart with animated data entry
- `AccuracyChart` — Area chart with gradient fill
- `FLRoundChart` — Bar chart showing per-round metrics
- `HospitalRadarChart` — Multi-axis radar for hospital comparison
- `PrivacyBudgetGauge` — Circular gauge showing ε consumption

**Forms (4 components):**
- `PatientDataForm` — 15+ field form for EHR input
- `ImageUploadForm` — File upload with preview
- `DrugScreenForm` — SMILES input with validation
- `TaskForm` — FL task configuration wizard

### 8.6 State Management

**Zustand Store (`useStore.ts`):**
- Global state for FL tasks, round metrics, and UI state
- Devtools integration for debugging
- Persist middleware for session recovery

**React Context:**
- `AuthContext.tsx` — JWT token, user info, login/logout functions
- `FLContext.tsx` — Active tasks, round history, WebSocket subscriptions

**Custom Hooks (6 hooks):**
- `useAuth` — Wraps AuthContext for component consumption
- `useFLStatus` — Polls task status with configurable interval
- `useHospitalNodes` — Tracks node health with auto-refresh
- `usePrediction` — Handles prediction API calls with loading states
- `useWebSocket` / `useWebSockets` — WebSocket connection management

### 8.7 Services Layer

All API calls are abstracted into service modules:
- `api.ts` — Base Axios instance with interceptors (JWT header, error handling)
- `fl.service.ts` — `createTask()`, `getTasks()`, `getTask()`, `startTask()`, `deleteTask()`
- `prediction.service.ts` — `predictEHR()`, `getHistory()`
- `imaging.service.ts` — `diagnoseImage()`, `getScans()`
- `drug.service.ts` — `screenDrug()`, `getResults()`
- `ids.service.ts` — `analyzePacket()`, `getThreats()`
- `socket.service.ts` — WebSocket connection lifecycle management

### 8.8 Design System

The UI uses a **glassmorphism** design language:
- Semi-transparent backgrounds with `backdrop-blur`
- Subtle border glow effects
- Smooth fade-in animations on page load
- Dark theme with carefully chosen accent colors
- Responsive layout that works on all screen sizes

---

## 📊 Phase 9 — Evaluation & IEEE Publication Plots

### 9.1 Global Evaluator (`evaluation/global_eval.py`)

Loads a saved model checkpoint and evaluates it on a global test set:
- Supports all 4 modules (EHR, Imaging, Drug, IDS)
- Computes comprehensive metrics per task
- Generates publication-ready plots

### 9.2 Metrics Engine (`evaluation/metrics.py`)

Computes standard machine learning metrics:
- **Accuracy** — Overall correct predictions / total predictions
- **Precision** — True positives / (True positives + False positives)
- **Recall (Sensitivity)** — True positives / (True positives + False negatives)
- **F1 Score** — Harmonic mean of Precision and Recall
- **AUC-ROC** — Area Under the Receiver Operating Characteristic Curve
- **Specificity** — True negatives / (True negatives + False positives)
- **Matthews Correlation Coefficient (MCC)** — Balanced metric for imbalanced datasets

### 9.3 IEEE Plotter (`evaluation/plotter.py`)

Generates publication-quality PDF plots formatted for IEEE conference submissions:
- **Font:** Times New Roman (IEEE standard)
- **Resolution:** 300 DPI (publication requirement)
- **Format:** PDF vector graphics
- **Plot types:**
  - ROC Curve with AUC annotation
  - Confusion Matrix with class labels
  - Loss convergence over federated rounds
  - Accuracy progression per hospital
  - Privacy budget consumption over time

### 9.4 Federated vs. Centralized Comparison (`evaluation/compare_federated_vs_central.py`)

A benchmark script that trains the same model in two ways and compares:
1. **Centralized:** All data pooled together (upper bound, but privacy-violating)
2. **Federated:** FedMedShield approach (privacy-preserving)

The comparison quantifies the "price of privacy" — how much accuracy we sacrifice for full privacy compliance. Results are saved to `evaluation/results/benchmark_comparison.json`.

---

## 🚀 Phase 10 — Deployment & Launch

### 10.1 PM2 Process Manager (`ecosystem.config.js`)

PM2 manages the backend process with:
- Auto-restart on crash
- File watching for hot-reload during development
- Log rotation and timestamping
- Environment variable injection

### 10.2 PowerShell Boot Script (`start_demo.ps1`)

A single-command launcher that:
1. Creates `logs/` directory if missing
2. Installs PM2 globally if not found
3. Starts the backend via PM2
4. Launches Vite frontend in a new PowerShell window
5. Prints access URLs and useful PM2 commands

### 10.3 Shell Setup Script (`setup.sh`)

Linux/Mac equivalent that:
1. Creates all project directories
2. Creates a Python virtual environment
3. Installs all pip dependencies
4. Installs all npm dependencies
5. Sets up environment variables

### 10.4 Python Orchestration (`run_all.py`)

An alternative launcher that:
1. Starts the FastAPI backend
2. Spawns the FL server
3. Spawns N hospital client processes
4. Starts the Vite frontend
5. Monitors all processes and restarts on failure

---

## ⚡ Quick Start Guide

### Prerequisites
- Python 3.9 or higher
- Node.js 18 or higher (with npm)
- Redis server (running on `localhost:6379`) — optional for demo mode
- PostgreSQL (running on `localhost:5432`) — optional for demo mode
- MongoDB (running on `localhost:27017`) — optional for demo mode

> **Note:** The dashboard includes a built-in simulation engine that works
> without any databases. You only need Python + Node.js for the demo.

### Step 1: Install Python Dependencies
```powershell
cd d:\FedCare
pip install -r backend/requirements.txt
```

### Step 2: Install Frontend Dependencies
```powershell
cd frontend
npm install
cd ..
```

### Step 3: Launch the System
```powershell
# Option A: PowerShell 1-click boot
.\start_demo.ps1

# Option B: Manual start
# Terminal 1 - Backend:
cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Frontend:
cd frontend && npm run dev
```

### Step 4: Open the Dashboard
Navigate to `http://localhost:5173` in your browser.

### Step 5: Run Training
Click **"Start EHR Task"** on the Dashboard to begin the real-time simulation.

---

## 📡 API Reference

### Authentication
| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| POST | `/api/auth/register` | `{ username, password }` | `{ user_id, token }` |
| POST | `/api/auth/login` | `{ username, password }` | `{ access_token, token_type }` |

### Federated Learning Tasks
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/tasks/` | Create a new FL task |
| GET | `/api/tasks/` | List all tasks |
| GET | `/api/tasks/{id}` | Get task details |
| POST | `/api/tasks/{id}/start` | Start training |
| DELETE | `/api/tasks/{id}` | Delete a task |

### Clinical AI Modules
| Method | Endpoint | Input | Output |
|--------|----------|-------|--------|
| POST | `/api/predict/ehr` | Patient vitals JSON | Risk prediction |
| POST | `/api/imaging/diagnose` | Medical image file | Diagnosis result |
| POST | `/api/drug/screen` | SMILES + protein target | Binding affinity |
| POST | `/api/ids/analyze` | Network packet features | Threat classification |

### WebSocket
| URL | Description |
|-----|-------------|
| `ws://localhost:8000/ws/metrics/{task_id}` | Real-time training metrics |

### Health Check
| Method | Endpoint | Response |
|--------|----------|----------|
| GET | `/` | Service info + module list |
| GET | `/health` | Database connection status |

---

## 🔧 Complete Tech Stack

### Backend
| Technology | Version | Role |
|-----------|---------|------|
| Python | 3.9+ | Runtime |
| FastAPI | 0.115 | Web framework |
| Uvicorn | 0.30 | ASGI server |
| Pydantic | 2.9 | Data validation |
| SQLAlchemy | 2.0 | PostgreSQL ORM |
| Motor (PyMongo) | 4.9 | MongoDB driver |
| Redis-py | 5.1 | Redis client |
| python-jose | 3.3 | JWT tokens |
| passlib | 1.7 | Password hashing |

### AI / Machine Learning
| Technology | Version | Role |
|-----------|---------|------|
| PyTorch | 2.4 | Deep learning framework |
| TorchVision | 0.19 | Image model utilities |
| Flower (flwr) | 1.11 | Federated learning framework |
| Scikit-learn | 1.5 | Classical ML metrics |
| NumPy | 1.26 | Numerical computing |
| Pandas | 2.2 | Data manipulation |
| OpenCV | 4.10 | Image processing |
| Pillow | 10.4 | Image I/O |

### Frontend
| Technology | Version | Role |
|-----------|---------|------|
| React | 18.3 | UI component library |
| Vite | 5.3 | Build tool + dev server |
| TypeScript | 5.2 | Type-safe JavaScript |
| TailwindCSS | 3.4 | Utility-first CSS |
| Recharts | 2.12 | Data visualization |
| Zustand | 4.5 | State management |
| Axios | 1.7 | HTTP client |
| React Router | 6.24 | Client routing |
| Lucide React | 0.395 | Icon library |

### Infrastructure
| Technology | Role |
|-----------|------|
| PostgreSQL | Relational data (users, tasks) |
| MongoDB | Document store (metrics, logs) |
| Redis | Pub/Sub + caching |
| PM2 | Process management |
| WebSockets | Real-time streaming |

---

## 📚 Research References

This project is grounded in peer-reviewed research:

1. **McMahan, B. et al.** (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS*. — Original FedAvg algorithm.

2. **Li, T. et al.** (2020). "Federated Optimization in Heterogeneous Networks." *MLSys*. — FedProx for Non-IID data.

3. **Wang, J. et al.** (2020). "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization." *NeurIPS*. — FedNova normalized averaging.

4. **Bonawitz, K. et al.** (2017). "Practical Secure Aggregation for Privacy-Preserving Machine Learning." *ACM CCS*. — Pairwise masking secure aggregation protocol.

5. **Abadi, M. et al.** (2016). "Deep Learning with Differential Privacy." *ACM CCS*. — DP-SGD algorithm for training neural networks with differential privacy.

6. **Mironov, I.** (2017). "Rényi Differential Privacy." *IEEE CSF*. — Tighter privacy accounting via Rényi Divergence.

7. **He, K. et al.** (2016). "Deep Residual Learning for Image Recognition." *CVPR*. — ResNet architecture used in Module 2.

8. **Lin, T.Y. et al.** (2017). "Focal Loss for Dense Object Detection." *ICCV*. — Focal Loss used in Module 4 for class imbalance.

---

## 📄 License & Data Provenance

This project is open-source software licensed under the **Apache License, Version 2.0**. See the [`LICENSE`](file:///d:/FedCare/LICENSE) file for complete terms and disclaimers.

### Dataset & Clinical Provenance Notice
- **Default Synthetic Data:** All default training, benchmarking, and demonstration pipelines execute on **synthetically generated data** designed for functional verification and architectural benchmarking.
- **External Datasets Not Bundled:** Real-world biomedical and network datasets (such as MIMIC-IV EHR records, BraTS MRI scans, Kaggle Glaucoma/Chest X-Ray archives, and NSL-KDD traffic logs) are **not bundled** with this repository.
- **Ethical & Regulatory Compliance:** Researchers supplying real patient or clinical data must ensure proper Institutional Review Board (IRB) oversight, execute required Data Use Agreements (DUAs), verify data provenance, and maintain compliance under HIPAA, GDPR, and local health information privacy frameworks.
- **Research Prototype Disclaimer:** FedMedShield is an academic research prototype. Outputs and telemetry are intended for evaluation and must not be used as clinical diagnostic recommendations or in active patient treatment.

---

<div align="center">

**Built with ❤️ for privacy-preserving healthcare AI**

*FedMedShield v2.0 — IEEE Federated Learning Systems*

</div>

