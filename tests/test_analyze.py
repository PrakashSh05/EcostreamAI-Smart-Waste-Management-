import asyncio
import os
import statistics
from pathlib import Path

import httpx


REQUEST_COUNT = int(os.getenv("ANALYZE_REQUEST_COUNT", "50"))
CONCURRENCY = int(os.getenv("ANALYZE_CONCURRENCY", "50"))
BASE_URL = os.getenv("ANALYZE_BASE_URL", "http://localhost:8000/analyze")
IMAGE_PATH = Path(os.getenv("ANALYZE_IMAGE_PATH", "image.png"))

# Stable coordinates to avoid geocoder variability in stress tests.
LATITUDE = os.getenv("ANALYZE_LAT", "12.9716")
LONGITUDE = os.getenv("ANALYZE_LON", "77.5946")
CITY = os.getenv("ANALYZE_CITY", "Bangalore")


def _percentile_95(values: list[float]) -> float:
	if not values:
		return 0.0
	sorted_vals = sorted(values)
	# Nearest-rank style percentile.
	idx = max(0, min(len(sorted_vals) - 1, int(round(0.95 * (len(sorted_vals) - 1)))))
	return float(sorted_vals[idx])


async def _single_request(client: httpx.AsyncClient, image_bytes: bytes, idx: int) -> dict:
	files = {"file": (f"stress_{idx}.png", image_bytes, "image/png")}
	data = {
		"latitude": LATITUDE,
		"longitude": LONGITUDE,
		"city": CITY,
	}

	response = await client.post(BASE_URL, data=data, files=files)
	response.raise_for_status()
	payload = response.json()
	timing = payload.get("timing_ms") or {}

	return {
		"yolo_ms": int(timing.get("yolo_ms", 0)),
		"rag_ms": int(timing.get("rag_ms", 0)),
		"db_ms": int(timing.get("db_ms", 0)),
		"total_ms": int(timing.get("total_ms", 0)),
	}


async def run_stress_test() -> None:
	if not IMAGE_PATH.exists():
		raise FileNotFoundError(f"Image payload not found: {IMAGE_PATH}")

	image_bytes = IMAGE_PATH.read_bytes()

	timeout = httpx.Timeout(120.0)
	limits = httpx.Limits(max_keepalive_connections=CONCURRENCY, max_connections=CONCURRENCY)
	semaphore = asyncio.Semaphore(CONCURRENCY)

	async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
		async def bounded_call(i: int):
			async with semaphore:
				return await _single_request(client, image_bytes, i)

		tasks = [asyncio.create_task(bounded_call(i)) for i in range(REQUEST_COUNT)]
		results = await asyncio.gather(*tasks, return_exceptions=True)

	errors = [r for r in results if isinstance(r, Exception)]
	successes = [r for r in results if isinstance(r, dict)]

	if errors:
		print(f"FAILED_REQUESTS: {len(errors)}")
		print(f"SUCCESSFUL_REQUESTS: {len(successes)}")
		print("FIRST_ERROR:", repr(errors[0]))

	if not successes:
		raise RuntimeError("No successful requests in stress test.")

	metrics = {}
	for key in ["yolo_ms", "rag_ms", "db_ms", "total_ms"]:
		vals = [float(item[key]) for item in successes]
		metrics[key] = {
			"avg": statistics.mean(vals),
			"p95": _percentile_95(vals),
		}

	print("=== Latency Stress Test (50 Concurrent Requests) ===")
	print(f"REQUEST_COUNT: {REQUEST_COUNT}")
	print(f"CONCURRENCY: {CONCURRENCY}")
	print(f"SUCCESSFUL_REQUESTS: {len(successes)}")
	print(f"FAILED_REQUESTS: {len(errors)}")
	print("")

	for key in ["yolo_ms", "rag_ms", "db_ms", "total_ms"]:
		print(f"{key}_avg_ms={metrics[key]['avg']:.2f}")
		print(f"{key}_p95_ms={metrics[key]['p95']:.2f}")

	total_kpi_met = metrics["total_ms"]["avg"] < 3000.0 and metrics["total_ms"]["p95"] < 3000.0
	print("")
	print(f"TOTAL_LATENCY_KPI_MET={str(total_kpi_met).upper()} (target: avg<3000 and p95<3000)")


if __name__ == "__main__":
	asyncio.run(run_stress_test())
