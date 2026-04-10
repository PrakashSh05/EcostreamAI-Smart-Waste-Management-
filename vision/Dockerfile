FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Install all deps including ultralytics (which pulls opencv-python)
RUN pip install --no-cache-dir --retries 3 --timeout 120 \
        ultralytics \
        fastapi \
        "uvicorn[standard]" \
        Pillow \
        numpy \
        python-multipart

# Step 2: Replace opencv-python with headless (no display server needed)
RUN pip uninstall -y opencv-python 2>/dev/null || true && \
    pip install --no-cache-dir opencv-python-headless

RUN mkdir -p /app/model

COPY serve.py /app/serve.py

ENV MODEL_PATH=/app/model/best.pt

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8001"]
