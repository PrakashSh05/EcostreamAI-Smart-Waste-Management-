FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn

COPY serve.py ./serve.py

EXPOSE 8001

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8001"]
