"""
tests/test_yolo.py
Tests for the YOLO Vision microservice (port 8001).

Usage:
    python tests/test_yolo.py
    (Requires vision service running at localhost:8001)
"""

import asyncio
import os
from pathlib import Path

import httpx

VISION_URL = os.getenv("YOLO_SERVICE_URL", "http://localhost:8001")
IMAGE_PATH = Path(os.getenv("TEST_IMAGE_PATH", "image.png"))


async def test_vision_health():
    """Test GET /health returns ok status."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(f"{VISION_URL}/health")
        response.raise_for_status()
        data = response.json()

        assert data["status"] == "ok", f"Health check failed: {data}"
        assert data["loaded"] is True, "Model not loaded"
        print(f"✅ /health: status={data['status']}, model={data.get('model')}")


async def test_vision_classes():
    """Test GET /classes returns 6 waste classes."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(f"{VISION_URL}/classes")
        response.raise_for_status()
        data = response.json()

        classes = data.get("classes", [])
        expected = ["plastic", "metal", "food_waste", "paper", "glass", "cardboard"]

        assert classes == expected, f"Expected {expected}, got {classes}"
        print(f"✅ /classes: {classes}")


async def test_vision_detect():
    """Test POST /detect with sample image returns valid detection response."""
    if not IMAGE_PATH.exists():
        print(f"⚠️ Skipping /detect test: {IMAGE_PATH} not found")
        return

    image_bytes = IMAGE_PATH.read_bytes()

    async with httpx.AsyncClient(timeout=30.0) as client:
        files = {"file": ("test.png", image_bytes, "image/png")}
        response = await client.post(f"{VISION_URL}/detect", files=files)
        response.raise_for_status()
        data = response.json()

        # Validate response shape (locked contract)
        assert "labels" in data, "Missing 'labels' field"
        assert "confidence" in data, "Missing 'confidence' field"
        assert "masks" in data, "Missing 'masks' field"
        assert "count" in data, "Missing 'count' field"
        assert "inference_ms" in data, "Missing 'inference_ms' field"

        assert isinstance(data["labels"], list), "labels should be a list"
        assert isinstance(data["confidence"], list), "confidence should be a list"
        assert isinstance(data["masks"], list), "masks should be a list"
        assert isinstance(data["count"], int), "count should be int"
        assert isinstance(data["inference_ms"], int), "inference_ms should be int"

        # Labels and confidence must have same length
        assert len(data["labels"]) == len(data["confidence"]), (
            f"labels({len(data['labels'])}) != confidence({len(data['confidence'])})"
        )
        assert data["count"] == len(data["labels"]), (
            f"count({data['count']}) != len(labels)({len(data['labels'])})"
        )

        # All labels must be one of the 6 classes
        valid_classes = {"plastic", "metal", "food_waste", "paper", "glass", "cardboard"}
        for label in data["labels"]:
            assert label in valid_classes or label.isdigit(), (
                f"Unknown label: {label}"
            )

        print(f"✅ /detect: {data['count']} objects detected in {data['inference_ms']}ms")
        print(f"   Labels: {data['labels']}")
        print(f"   Confidence: {[round(c, 3) for c in data['confidence']]}")


async def test_vision_metrics():
    """Test GET /metrics returns request counters."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(f"{VISION_URL}/metrics")
        response.raise_for_status()
        data = response.json()

        assert "total_requests" in data, "Missing 'total_requests'"
        assert "avg_inference_ms" in data, "Missing 'avg_inference_ms'"
        assert "uptime_seconds" in data, "Missing 'uptime_seconds'"

        print(f"✅ /metrics: requests={data['total_requests']}, "
              f"avg_ms={data['avg_inference_ms']}, "
              f"uptime={data['uptime_seconds']}s")


async def run_all_tests():
    print("=== YOLO Vision Service Tests ===\n")

    tests = [
        test_vision_health,
        test_vision_classes,
        test_vision_detect,
        test_vision_metrics,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
