# EcoStream AI

## Quick Start (New Teammates)
1. Clone and enter project folder.
2. Create .env from .env.example and fill values.
3. Start services:
	 docker compose up --build -d
4. Build local vector store:
	 docker exec ecostream_backend python rag/ingest.py
5. Verify API health:
	 http://localhost:8000/health

## Setup
1. Clone the repo.
2. Copy .env.example to .env and fill in your values.
3. Start services:
	 docker compose up --build -d
4. Verify backend health:
	 http://localhost:8000/health
5. Build local vector store from TXT corpus:
	 docker exec ecostream_backend python rag/ingest.py

## Corpus and Vector Store Policy
- TXT files inside pdfs are pushed to git and treated as the source corpus.
- Non-TXT files inside pdfs are ignored.
- chroma_store is not pushed to git.
- Running the ingest command creates and populates chroma_store automatically on each machine.

## Database Initialization
- No manual init script is required now.
- The database schema is created automatically from backend/db/init.sql when the Postgres volume is created for the first time.
- If you need a clean reset:
	1. docker compose down -v
	2. docker compose up --build -d

## Post-Push Update Steps (All Members)
1. Save local work and commit or stash any local changes.
2. Pull latest changes from your branch or main.
3. Rebuild and restart containers:
	 docker compose up --build -d
4. Verify services are running:
	 docker ps
5. Rebuild local vector store from latest TXT corpus:
	 docker exec ecostream_backend python rag/ingest.py
6. Verify backend responds:
	 open http://localhost:8000/health

## Member-Wise Action Checklist

### Member 1 (Vision)
- Confirm vision service starts successfully after rebuild.
- Run one analyze request using image.png and verify detected_materials is populated.
- Check that labels remain compatible with backend and frontend expectations.

### Member 2 (RAG/LLM)
- Rebuild vector store after pulling corpus or RAG changes:
	docker exec ecostream_backend python rag/ingest.py
- Run quality evaluation:
	docker exec ecostream_backend python tests/test_rag.py
- Confirm latest KPI target remains satisfied in output:
	faithfulness > 0.85 and answer_relevancy > 0.85.
- If retrieval behavior changes, re-check mixed household waste grounding first.

### Member 3 (Backend)
- Confirm ingest job completed at least once after latest pull.
- Run latency stress test:
	docker exec ecostream_backend python tests/test_analyze.py
- Confirm total_ms KPI under 50 concurrent requests remains below 3000ms.
- Re-check /health and /analyze endpoints for 200 responses.

### Member 4 (Frontend)
- Pull latest backend-compatible changes and start frontend:
	npm install
	npm run dev
- Verify dashboard and citizen flows still work with current /analyze response fields:
	detected_materials, disposal_advice, timing_ms.

### Member 5 (Analytics/Paper)
- Use metrics/final_paper_results.txt as the source for paper tables.
- Include both average and P95 latency in performance analysis.
- Include current retrieval settings from rag/config.py in experimental setup.

## Branches
- main — stable, reviewed code only
- feature/vision — Member 1
- feature/rag-llm — Member 2 (you)
- feature/backend — Member 3 (you)
- feature/frontend — Member 4
- feature/analytics — Member 5