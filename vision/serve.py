"""
vision/serve.py
FastAPI microservice that exposes YOLOv11s-seg inference on port 8001.

Endpoints
---------
POST /detect        — run inference, return labels/confidence/masks/count/inference_ms
GET  /health        — liveness probe
GET  /classes       — class list
GET  /metrics       — request counters and uptime

Member 3's yolo_client.py calls POST /detect and reads the "labels" field.
The full response shape is locked — do not change field names.

Usage:
    uvicorn vision.serve:app --host 0.0.0.0 --port 8001
    python vision/serve.py

Environment variables:
    MODEL_PATH           Path to best.pt  (default: vision/model/best.pt)
    YOLO_CONF_THRESHOLD  Confidence threshold (default: 0.35)
    YOLO_IOU_THRESHOLD   IOU threshold (default: 0.45)

Dependencies:
    fastapi, uvicorn, ultralytics, Pillow, numpy, python-multipart
"""

import io
import logging
import os
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH     = Path(os.getenv("MODEL_PATH", "vision/model/best.pt"))
CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.35"))
IOU_THRESHOLD  = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))

CLASS_NAMES = ["plastic", "metal", "food_waste", "paper", "glass", "cardboard"]

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_model: YOLO | None = None
_total_requests: int = 0
_cumulative_inference_ms: float = 0.0
_startup_time: float = time.time()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EcoStream AI Vision Service",
    description="YOLOv11s-seg instance segmentation for waste material detection.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Startup: load model and warm up
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def load_model() -> None:
    global _model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"best.pt not found at {MODEL_PATH}. Run vision/train.py first."
        )
    log.info("Loading YOLO model from %s", MODEL_PATH)
    _model = YOLO(str(MODEL_PATH))
    # Use the model's own class names instead of hardcoding
    global CLASS_NAMES
    if hasattr(_model, 'names') and _model.names:
        CLASS_NAMES = [_model.names[i] for i in sorted(_model.names.keys())]
        log.info("Using model's class names: %s", CLASS_NAMES)
    # Warm-up: one blank inference so the first real request is not penalised
    _model.predict(
        np.zeros((640, 640, 3), dtype=np.uint8),
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
    )
    log.info(
        "Model ready — conf=%.2f  iou=%.2f  classes=%s",
        CONF_THRESHOLD, IOU_THRESHOLD, CLASS_NAMES,
    )


# ---------------------------------------------------------------------------
# POST /detect
# ---------------------------------------------------------------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> JSONResponse:
    """
    Run YOLOv11s-seg on an uploaded image.

    Request:  multipart/form-data, field 'file' (JPEG or PNG)
    Response (shape is locked — do not change field names):
    {
        "labels":       ["plastic", "metal"],
        "confidence":   [0.91, 0.87],
        "masks":        [[[0.12, 0.34], [0.15, 0.40]], [[0.55, 0.60]]],
        "count":        2,
        "inference_ms": 423
    }
    masks: normalised polygon [x,y] pairs (0.0-1.0).
           Empty list [] for any detection where mask is unavailable.
    """
    global _total_requests, _cumulative_inference_ms

    _total_requests += 1

    # ---- decode image -------------------------------------------------------
    try:
        raw = await file.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        log.warning("Invalid image upload: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_image", "detail": str(exc)},
        )

    img_w, img_h = pil_img.size
    img_np = np.array(pil_img)

    # ---- inference (measure only YOLO time) ---------------------------------
    t0 = time.perf_counter()
    try:
        results = _model.predict(
            source=img_np,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )
    except Exception as exc:
        log.error("Inference failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "inference_failed", "detail": str(exc)},
        )
    inference_ms = int((time.perf_counter() - t0) * 1000)
    _cumulative_inference_ms += inference_ms

    # ---- parse result -------------------------------------------------------
    r = results[0]
    labels: list[str]              = []
    confidence: list[float]        = []
    masks: list[list[list[float]]] = []

    if r.boxes is not None and len(r.boxes) > 0:
        cls_ids   = r.boxes.cls.cpu().tolist()
        confs     = r.boxes.conf.cpu().tolist()
        log.info(
            "Raw detections: %d objects — cls_ids=%s, confs=%s",
            len(cls_ids), cls_ids, [round(c, 3) for c in confs],
        )
        has_masks = r.masks is not None
        # masks.xyn: normalised polygon coords already in 0-1 range
        masks_xyn = r.masks.xyn if has_masks else [None] * len(cls_ids)

        for idx, (cls_id, conf) in enumerate(zip(cls_ids, confs)):
            cls_id = int(cls_id)
            labels.append(
                CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            )
            confidence.append(round(float(conf), 4))

            poly_raw = masks_xyn[idx] if idx < len(masks_xyn) else None
            if poly_raw is not None and len(poly_raw) > 0:
                masks.append([
                    [round(float(x), 6), round(float(y), 6)]
                    for x, y in poly_raw
                ])
            else:
                masks.append([])

    count = len(labels)
    log.info(
        "detect | %s | %dx%d | %d object(s) | %dms",
        file.filename or "upload", img_w, img_h, count, inference_ms,
    )

    return JSONResponse(content={
        "labels":       labels,
        "confidence":   confidence,
        "masks":        masks,
        "count":        count,
        "inference_ms": inference_ms,
    })


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "model":  "yolov11-seg",
        "loaded": _model is not None,
    })


# ---------------------------------------------------------------------------
# GET /classes
# ---------------------------------------------------------------------------
@app.get("/classes")
async def classes() -> JSONResponse:
    return JSONResponse({"classes": CLASS_NAMES})


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------
@app.get("/metrics")
async def metrics() -> JSONResponse:
    avg_ms = (
        round(_cumulative_inference_ms / _total_requests, 1)
        if _total_requests > 0 else 0.0
    )
    return JSONResponse({
        "total_requests":   _total_requests,
        "avg_inference_ms": avg_ms,
        "uptime_seconds":   round(time.time() - _startup_time, 1),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("vision.serve:app", host="0.0.0.0", port=8001, reload=False)
