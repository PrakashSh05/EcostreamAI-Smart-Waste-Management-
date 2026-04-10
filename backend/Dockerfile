FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies required for ChromaDB and PostgreSQL
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy and install requirements first (to cache this layer and speed up future builds)
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 3 --timeout 120 -r requirements.txt

# 3. Explicitly copy only the necessary folders to avoid image bloat
COPY backend/ ./backend/
COPY rag/ ./rag/
COPY analytics/ ./analytics/

# 4. Your brilliant addition to fix Python import paths
ENV PYTHONPATH=/app

# 5. Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]