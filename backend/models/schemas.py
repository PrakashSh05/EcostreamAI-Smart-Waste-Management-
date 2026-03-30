from typing import List

from pydantic import BaseModel


class AnalyzeLocation(BaseModel):
	lat: float
	lng: float


class AnalyzeTimingMs(BaseModel):
	yolo_ms: int
	rag_ms: int
	db_ms: int
	total_ms: int


class AnalyzeResponse(BaseModel):
	scan_id: str
	detected_materials: List[str]
	disposal_advice: str
	timestamp: str
	location: AnalyzeLocation
	timing_ms: AnalyzeTimingMs
