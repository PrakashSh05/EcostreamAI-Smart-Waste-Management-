import os
import time
import uuid
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, UploadFile

from backend.db.postgres import get_connection
from backend.models.schemas import AnalyzeLocation, AnalyzeResponse, AnalyzeTimingMs
from backend.services import yolo_client
from backend.services.vision_classifier import classify_waste_image
from backend.services.geocode import get_city_from_coords
from rag.query import get_disposal_advice

router = APIRouter(tags=["analyze"])
logger = logging.getLogger(__name__)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
	file: UploadFile = File(...),
	latitude: float = Form(...),
	longitude: float = Form(...),
	city: str | None = Form(None),
):
	overall_start = time.time()

	image_bytes = await file.read()

	# Step 1: Run YOLO detection (for object detection + bounding boxes)
	yolo_start = time.time()
	yolo_result = await yolo_client.detect_materials(image_bytes)
	yolo_ms = int((time.time() - yolo_start) * 1000)

	if isinstance(yolo_result, dict):
		yolo_labels = yolo_result.get("labels") or yolo_result.get("materials") or []
	else:
		yolo_labels = yolo_result or []
	yolo_labels = [str(item) for item in yolo_labels]

	# Step 2: Run Groq Vision classifier (for accurate material classification)
	vision_start = time.time()
	try:
		vision_labels = await classify_waste_image(image_bytes)
	except Exception as e:
		logger.warning("Vision classifier failed, using YOLO only: %s", e)
		vision_labels = []
	vision_ms = int((time.time() - vision_start) * 1000)

	# Step 3: Merge results — prefer Vision classifier, fall back to YOLO
	if vision_labels:
		labels = vision_labels
		logger.info(
			"Using Vision classifier: %s (YOLO said: %s)",
			vision_labels, yolo_labels,
		)
	else:
		labels = yolo_labels
		logger.info("Using YOLO labels: %s", yolo_labels)

	# Deduplicate
	labels = list(dict.fromkeys(labels))

	resolved_city = get_city_from_coords(latitude, longitude) if not city else city

	rag_start = time.time()
	advice = get_disposal_advice(materials=tuple(labels), city=resolved_city)
	rag_ms = int((time.time() - rag_start) * 1000)

	scan_id = str(uuid.uuid4())
	created_at = datetime.now(timezone.utc)

	db_start = time.time()
	conn = get_connection()
	try:
		with conn.cursor() as cur:
			cur.execute(
				"""
				INSERT INTO scans (
					id,
					created_at,
					latitude,
					longitude,
					materials,
					advice,
					city,
					yolo_ms,
					rag_ms,
					db_ms,
					total_ms,
					is_collected
				) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE)
				""",
				(
					scan_id,
					created_at,
					latitude,
					longitude,
					labels,
					advice,
					resolved_city,
					yolo_ms,
					rag_ms,
					0,
					0,
				),
			)
			conn.commit()
	finally:
		conn.close()

	db_ms = int((time.time() - db_start) * 1000)
	total_ms = int((time.time() - overall_start) * 1000)

	conn = get_connection()
	try:
		with conn.cursor() as cur:
			cur.execute(
				"""
				UPDATE scans
				SET db_ms = %s, total_ms = %s
				WHERE id = %s
				""",
				(db_ms, total_ms, scan_id),
			)
			conn.commit()
	finally:
		conn.close()

	return AnalyzeResponse(
		scan_id=scan_id,
		detected_materials=labels,
		disposal_advice=advice,
		timestamp=created_at.isoformat(),
		location=AnalyzeLocation(lat=latitude, lng=longitude),
		timing_ms=AnalyzeTimingMs(
			yolo_ms=yolo_ms,
			rag_ms=rag_ms,
			db_ms=db_ms,
			total_ms=total_ms,
		),
	)
