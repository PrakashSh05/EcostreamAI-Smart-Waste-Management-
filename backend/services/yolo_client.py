import os
import httpx
from fastapi import HTTPException

YOLO_SERVICE_URL = os.getenv("YOLO_SERVICE_URL", "http://vision:8001")

async def detect_materials(image_bytes: bytes) -> list[str]:
    try:
        async with httpx.AsyncClient() as client:
            files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
            response = await client.post(f"{YOLO_SERVICE_URL}/detect", files=files, timeout=15.0)
            response.raise_for_status()
            data = response.json()
            return data.get("labels", [])
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vision service unavailable: {str(e)}")
