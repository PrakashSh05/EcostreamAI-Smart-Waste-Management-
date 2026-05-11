<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv11s--seg-Ultralytics-FF6F00?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-RAG-green?logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white" />
</p>

<h1 align="center">♻️ EcoStream AI</h1>
<h3 align="center">AI-Powered Smart Waste Management Platform</h3>

<p align="center">
  <b>Scan · Classify · Advise · Predict</b><br/>
  An end-to-end system that detects waste materials via computer vision, provides city-specific disposal advice through a RAG-powered LLM, and predicts next-day waste hotspots for municipal operators.
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Performance Metrics](#-performance-metrics)
- [Testing & Evaluation](#-testing--evaluation)
- [Team Contributions](#-team-contributions)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🌍 Overview

**EcoStream AI** is a full-stack, AI-powered waste management platform built to address urban waste segregation and disposal challenges in Indian cities. The system combines three AI capabilities into a single, Docker-orchestrated platform:

| Capability | What It Does |
|:---|:---|
| 🔍 **Computer Vision** | YOLOv11s-seg model trained on 6,467 synthetic images detects and segments 6 waste material classes in real-time |
| 🧠 **RAG Advisory** | Hybrid retrieval pipeline (dense + sparse + reranking) grounded in India's SWM Rules 2016 provides city-specific disposal guidance |
| 📊 **Predictive Analytics** | Time-weighted KDE and exponential decay forecasting predicts next-day waste hotspots for municipal operators |

**Supported Cities:** Bangalore · Mumbai · Delhi · Chennai

---

## ✨ Key Features

### 👤 Citizen Portal
- **Live Camera Scan** — Point your phone camera at waste, get instant material classification
- **AI Chatbot** — Ask natural-language waste disposal questions with optional image upload
- **Client-side image compression** — Optimized uploads for mobile networks

### 🏛️ Government Dashboard
- **Real-time Heatmap** — Leaflet.js map with color-coded waste density (Green/Orange/Red)
- **Predictive Overlay** — Toggle tomorrow's predicted hotspots (purple markers)
- **Zone Resolution** — Mark zones as "collected" to clear the heatmap
- **Top Materials Ranking** — Live statistics of most detected waste types per city

### 🔧 Engineering
- **Microservice Architecture** — 4 Docker Compose services with health checks
- **Hybrid RAG Retrieval** — BGE embeddings + BM25 sparse search + RRF fusion + cross-encoder reranking
- **Dual-classifier Pipeline** — YOLO detection + Groq Vision classifier with automatic fallback
- **Per-request Instrumentation** — Every API call logs YOLO/RAG/DB/total latency to PostgreSQL

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CITIZEN / OPERATOR                          │
│                     (React Frontend — Vite)                        │
│         ┌──────────────┐          ┌───────────────────┐            │
│         │ Citizen Portal│          │  Gov Dashboard    │            │
│         │ • LiveScan    │          │  • HeatMap        │            │
│         │ • ChatBot     │          │  • PredictiveLayer│            │
│         │ • FloatingChat│          │  • Zone Resolve   │            │
│         └──────┬───────┘          └────────┬──────────┘            │
└────────────────┼───────────────────────────┼──────────────────────-┘
                 │  HTTP (axios)             │
                 ▼                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI :8000)                          │
│                                                                     │
│  POST /analyze ─── YOLO Client ──────► Vision Service (:8001)      │
│       │                                  YOLOv11s-seg              │
│       ├── Groq Vision Classifier (fallback)                        │
│       ├── RAG Pipeline ──► ChromaDB + Groq LLM                    │
│       └── PostgreSQL Write (scan log)                              │
│                                                                     │
│  POST /chat ───── RAG Pipeline ──► LangChain → Groq Llama-3.1-8B  │
│  GET  /heatmap ── KDE Engine ──► PostgreSQL (uncollected scans)    │
│  GET  /predict ── Prediction Engine ──► PostgreSQL (14-day history)│
│  POST /scans/resolve ── Mark zone collected ──► PostgreSQL         │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                      │
│                                                                     │
│  PostgreSQL 15 ─── scans table (UUID, lat/lng, materials, timing)  │
│  ChromaDB ──────── Vector store (SWM Rules corpus, BGE embeddings) │
│  pdfs/ ─────────── Regulatory text corpus (5 city documents)       │
└─────────────────────────────────────────────────────────────────────┘
```

### Request Flow: Citizen Scans Waste

```
Camera Frame → Client Compression → POST /analyze
  → YOLO Detection (489ms avg) → Material Labels
  → Groq Vision Classifier (fallback)
  → RAG: Hybrid Retrieve → Rerank → LLM Generate (376ms avg)
  → PostgreSQL: Log scan + latency (16ms avg)
  → Response: { materials, advice, timing_ms } (979ms total)
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Frontend** | React 19, Vite 8, React Router 7 | SPA with citizen + dashboard views |
| **Maps** | Leaflet.js, react-leaflet | Heatmap and predictive overlay |
| **Backend** | FastAPI 0.104, Uvicorn | REST API orchestrator |
| **Vision** | YOLOv11s-seg (Ultralytics), Groq Vision | Waste detection + classification |
| **RAG** | LangChain, ChromaDB, Groq (Llama-3.1-8B) | Disposal advice generation |
| **Embeddings** | BAAI/bge-base-en-v1.5 (HuggingFace) | Dense vector retrieval |
| **Sparse Search** | BM25Okapi (rank-bm25) | Keyword-based retrieval |
| **Reranking** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-encoder reranking |
| **Analytics** | scikit-learn (KDE), NumPy | Heatmap + hotspot prediction |
| **Database** | PostgreSQL 15 (Alpine) | Scan persistence + spatial queries |
| **Dataset** | TACO + TrashNet (GAN-composited) | 6,467 synthetic training images |
| **Evaluation** | RAGAS, httpx (async stress tests) | RAG quality + latency benchmarks |
| **DevOps** | Docker Compose, multi-stage Dockerfiles | 4-service orchestration |
| **Research** | LaTeX (IEEE format) | Academic paper |

---

## 📁 Project Structure

```
EcostreamAI-Smart-Waste-Management/
│
├── backend/                    # FastAPI backend service
│   ├── Dockerfile              # Backend container (Python 3.11)
│   ├── main.py                 # App entry — mounts all routers
│   ├── db/
│   │   ├── init.sql            # PostgreSQL schema + partial indexes
│   │   └── postgres.py         # Connection pool manager
│   ├── routes/
│   │   ├── analyze.py          # POST /analyze — main scan pipeline
│   │   ├── chat.py             # POST /chat — AI chatbot endpoint
│   │   ├── health.py           # GET /health — liveness probe
│   │   └── scans.py            # GET /scans, POST /scans/resolve
│   ├── models/
│   │   └── schemas.py          # Pydantic response models
│   └── services/
│       ├── yolo_client.py      # HTTP client → Vision service
│       ├── vision_classifier.py# Groq Vision fallback classifier
│       └── geocode.py          # Lat/lng → city resolver
│
├── vision/                     # Computer Vision microservice
│   ├── Dockerfile              # Vision container with Ultralytics
│   ├── serve.py                # FastAPI server (POST /detect, :8001)
│   ├── train.py                # YOLOv11s-seg training script
│   └── model/
│       ├── best.pt             # Trained weights (seg model)
│       ├── baseline.pt         # YOLOv8n-det baseline for ablation
│       └── runs/               # Training logs, curves, confusion matrices
│
├── rag/                        # RAG pipeline
│   ├── config.py               # chunk_size=500, k=3, BGE model config
│   ├── ingest.py               # Corpus → ChromaDB vector store builder
│   ├── query.py                # Hybrid retrieval + LLM generation (780 LOC)
│   └── prompts.py              # System prompt with strict grounding rules
│
├── analytics/                  # Predictive analytics module
│   ├── kde.py                  # Gaussian KDE heatmap (grid + sklearn)
│   ├── predict.py              # Next-day hotspot prediction engine
│   └── routes/
│       └── predict.py          # GET /heatmap, GET /predict endpoints
│
├── dataset/                    # Data pipeline
│   └── gan_mix.py              # TACO + TrashNet → synthetic YOLO dataset
│
├── frontend/                   # React SPA
│   ├── index.html              # Entry point
│   ├── package.json            # React 19, Vite 8, Leaflet, axios
│   └── src/
│       ├── App.jsx             # Router: /citizen, /dashboard
│       ├── index.css           # Complete design system (22KB)
│       ├── api/
│       │   └── client.js       # API client (analyze, chat, heatmap, predict)
│       ├── components/
│       │   ├── LiveScan.jsx    # Camera capture + compression + scan
│       │   ├── ChatBot.jsx     # Scan result display
│       │   ├── FloatingChat.jsx# Floating AI assistant
│       │   ├── HeatMap.jsx     # Leaflet heatmap with CircleMarkers
│       │   └── PredictiveLayer.jsx  # Tomorrow's prediction overlay
│       └── pages/
│           ├── CitizenPortal.jsx    # Citizen scan + chatbot page
│           └── GovDashboard.jsx     # Operator command center
│
├── pdfs/                       # Regulatory text corpus
│   ├── SWM_Rules_2016.txt      # National Solid Waste Management Rules
│   ├── BBMP_Segregation.txt    # Bangalore municipal rules
│   ├── MCGM_Mumbai_Rules.txt   # Mumbai municipal rules
│   ├── NDMC_Delhi_Rules.txt    # Delhi municipal rules
│   └── GCC_Chennai_Rules.txt   # Chennai municipal rules
│
├── tests/                      # Evaluation suite
│   ├── test_rag.py             # RAGAS evaluation (faithfulness + relevancy)
│   ├── test_analyze.py         # Concurrent latency stress test (50 reqs)
│   ├── test_predict.py         # Analytics endpoint tests
│   ├── test_yolo.py            # Vision model evaluation
│   ├── independent_eval.py     # 4-section, 400-point scoring suite
│   ├── rag_ground_truth.json   # 15 ground-truth QA pairs
│   └── ablation_results.json   # YOLOv11s-seg vs YOLOv8n-det comparison
│
├── metrics/                    # Performance results
│   ├── final_paper_results.txt # RAG + latency metrics for IEEE paper
│   ├── RAGAS test result.png   # RAGAS evaluation screenshot
│   └── Latenct Test (50 concurrent Requests).png
│
├── research/                   # IEEE research paper
│   ├── paper.tex               # Main LaTeX document
│   ├── references.bib          # Bibliography
│   └── sections/               # Paper sections
│
├── scripts/                    # Deployment & setup scripts
│   ├── deploy.sh               # Production deployment
│   ├── ingest_pdfs.sh          # Vector store rebuild
│   └── init_db.sh              # Database initialization
│
├── docker-compose.yml          # 4-service orchestration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- **Docker** & **Docker Compose** (v2+)
- **Node.js** 18+ (for frontend development)
- **Groq API Key** — [Get one free at groq.com](https://console.groq.com)

### 1. Clone & Configure

```bash
git clone https://github.com/PrakashSh05/EcostreamAI-Smart-Waste-Management-.git
cd EcostreamAI-Smart-Waste-Management-

# Create environment file
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 2. Start All Services

```bash
docker compose up --build -d
```

This starts 4 containers:

| Container | Port | Service |
|:---|:---|:---|
| `ecostream_db` | 5432 | PostgreSQL 15 (auto-creates schema) |
| `ecostream_backend` | 8000 | FastAPI backend + RAG pipeline |
| `ecostream-vision` | 8001 | YOLOv11s-seg inference API |
| Frontend (dev) | 5173 | React development server |

### 3. Build the Vector Store

```bash
docker exec ecostream_backend python rag/ingest.py
```

This processes the 5 regulatory text documents in `pdfs/` into ChromaDB embeddings.

### 4. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

### 5. Verify Everything Works

```bash
# Backend health
curl http://localhost:8000/health
# → {"status": "ok"}

# Vision health
curl http://localhost:8001/health
# → {"status": "ok", "model": "yolov11-seg", "loaded": true}

# Test a scan
curl -X POST http://localhost:8000/analyze \
  -F "file=@image.png" \
  -F "latitude=12.9716" \
  -F "longitude=77.5946" \
  -F "city=Bangalore"
```

### Database Reset (if needed)

```bash
docker compose down -v        # Remove volumes
docker compose up --build -d  # Recreate from init.sql
docker exec ecostream_backend python rag/ingest.py  # Rebuild vectors
```

---

## 📡 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/analyze` | Upload image → detect materials → get disposal advice |
| `POST` | `/chat` | Natural-language waste question (optional image) |
| `GET` | `/health` | Backend liveness probe |
| `GET` | `/heatmap?city=Bangalore` | KDE heatmap of uncollected waste |
| `GET` | `/predict?city=Bangalore` | Next-day hotspot predictions |
| `GET` | `/scans?city=Bangalore` | List uncollected scans |
| `POST` | `/scans/resolve` | Mark zone as collected |

### Vision Microservice (port 8001)

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/detect` | Run YOLOv11s-seg inference on image |
| `GET` | `/health` | Vision service liveness |
| `GET` | `/classes` | List 6 waste classes |
| `GET` | `/metrics` | Request count, avg inference time, uptime |

### POST /analyze — Response Shape

```json
{
  "scan_id": "uuid",
  "detected_materials": ["plastic", "food_waste"],
  "disposal_advice": "In Bangalore, plastic waste should be placed in the blue/dry waste bin...",
  "timestamp": "2026-04-10T12:00:00+00:00",
  "location": { "lat": 12.9716, "lng": 77.5946 },
  "timing_ms": {
    "yolo_ms": 489,
    "rag_ms": 376,
    "db_ms": 16,
    "total_ms": 979
  }
}
```

---

## 📊 Performance Metrics

All metrics measured on the deployed Docker Compose stack.

### RAG Quality (RAGAS Evaluation — 15 test cases)

| Metric | Score | Target | Status |
|:---|:---|:---|:---|
| **Faithfulness** | 0.9222 | > 0.85 | ✅ PASS |
| **Answer Relevancy** | 0.8607 | > 0.85 | ✅ PASS |

> Evaluated using `llama-3.3-70b-versatile` as the RAGAS judge model against 15 ground-truth QA pairs covering all 4 cities.

### Latency (50 Concurrent Requests)

| Component | Avg Latency | P95 Latency |
|:---|:---|:---|
| YOLO Detection | 489.20 ms | 796.00 ms |
| RAG Generation | 375.74 ms | — |
| Database Write | 15.92 ms | 17.00 ms |
| **End-to-End** | **978.86 ms** | **966.00 ms** |

> **KPI Target: < 3,000ms** → Achieved **979ms avg** (3.1× under budget) ✅

### Vision Model (Ablation Study)

| Model | Box mAP50-95 | Mask mAP50-95 | Dataset |
|:---|:---|:---|:---|
| YOLOv8n-det (baseline) | 0.6219 | — | 6,467 images |
| **YOLOv11s-seg** | **0.6503** (+2.8pp) | **0.5533** | 6,467 images |

> Segmentation model provides pixel-level instance masks that detection models fundamentally cannot offer.

### RAG Configuration

| Parameter | Value |
|:---|:---|
| Embedding Model | `BAAI/bge-base-en-v1.5` |
| LLM | `llama-3.1-8b-instant` (via Groq) |
| Chunk Size / Overlap | 500 / 50 |
| Retriever K | 3 |
| Retrieval Strategy | Hybrid (Dense + BM25 + RRF + Cross-Encoder) |

---

## 🧪 Testing & Evaluation

### Run RAG Quality Tests

```bash
docker exec ecostream_backend python tests/test_rag.py
```

Runs RAGAS evaluation with faithfulness and answer relevancy metrics. Requires `GROQ_API_KEY`.

### Run Latency Stress Test

```bash
docker exec ecostream_backend python tests/test_analyze.py
```

Fires 50 concurrent requests and reports avg/P95 latency per component.

### Run Analytics Endpoint Tests

```bash
python tests/test_predict.py
```

Tests `/heatmap` and `/predict` endpoints for correctness.

### Run Independent Evaluation Suite

```bash
python tests/independent_eval.py
```

400-point scoring rubric across 4 sections: Vision Model, Backend API, Frontend, Vision Microservice.

---

## 👥 Team Contributions

| Member | Role | Key Deliverables |
|:---|:---|:---|
| **Member 1** | Vision Engineer | YOLOv11s-seg training, `vision/serve.py`, Dockerfile, 6-class model |
| **Member 2** | RAG/LLM Engineer | Hybrid retrieval pipeline, `rag/query.py` (780 LOC), prompt engineering |
| **Member 3** | Backend Engineer | FastAPI orchestrator, PostgreSQL schema, Docker Compose, YOLO client |
| **Member 4** | Frontend Engineer | React SPA, Citizen Portal, Gov Dashboard, Leaflet heatmap |
| **Member 5** | Analytics & Evaluation | KDE heatmap, prediction engine, RAGAS evaluation, stress tests, IEEE paper |

---

## 🔮 Future Work

- [ ] **DBSCAN Clustering** — Replace fixed 500m grid with variable-density hotspot detection
- [ ] **TimescaleDB Migration** — Time-series optimized storage for scan history
- [ ] **Online Learning** — Operator feedback (zone resolved) retrains prediction weights
- [ ] **Multi-language Support** — Disposal advice in Hindi, Kannada, Tamil, Marathi
- [ ] **Mobile App** — React Native wrapper with offline-first camera scanning
- [ ] **Scheduled Collection Routes** — TSP-based route optimization for waste trucks
- [ ] **Real-time WebSocket Updates** — Live heatmap refresh without polling
- [ ] **Model Quantization** — INT8 YOLO model for edge deployment on Raspberry Pi
- [ ] **Expanded Corpus** — Add waste management rules for 20+ Indian cities
- [ ] **User Authentication** — Role-based access (citizen vs operator vs admin)

---

## 📄 Environment Variables

Copy `.env.example` to `.env` and configure:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# PostgreSQL (defaults work with Docker Compose)
POSTGRES_DB=ecostream
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_HOST=db
POSTGRES_PORT=5432

# Service URLs
YOLO_SERVICE_URL=http://vision:8001
CHROMA_PERSIST_DIR=/app/chroma_store
```

---

## 📜 License

This project was developed as an academic project (PCL coursework). All rights reserved by the contributors.

---

<p align="center">
  Built with ❤️ by the EcoStream AI Team
</p>