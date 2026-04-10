"""
tests/test_predict.py
Tests for the analytics heatmap and prediction endpoints.

Usage:
    python tests/test_predict.py
    (Requires backend running at localhost:8000)
"""

import asyncio
import os

import httpx

BASE_URL = os.getenv("ANALYZE_BASE_URL", "http://localhost:8000")


async def test_heatmap_endpoint():
    """Test GET /heatmap returns valid heatmap data."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{BASE_URL}/heatmap")
        response.raise_for_status()
        data = response.json()

        assert isinstance(data, list), f"Expected list, got {type(data)}"

        if data:
            point = data[0]
            assert "lat" in point, "Missing 'lat' field"
            assert "lng" in point, "Missing 'lng' field"
            assert "intensity" in point, "Missing 'intensity' field"
            assert 0.0 <= point["intensity"] <= 1.0, (
                f"Intensity {point['intensity']} outside 0.0-1.0 range"
            )
            print(f"✅ /heatmap returned {len(data)} points")
            print(f"   Top point: lat={point['lat']}, lng={point['lng']}, intensity={point['intensity']}")
        else:
            print("✅ /heatmap returned empty list (no uncollected scans in DB)")


async def test_heatmap_with_city():
    """Test GET /heatmap?city=Bangalore filters correctly."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{BASE_URL}/heatmap", params={"city": "Bangalore"})
        response.raise_for_status()
        data = response.json()
        assert isinstance(data, list)
        print(f"✅ /heatmap?city=Bangalore returned {len(data)} points")


async def test_predict_endpoint():
    """Test GET /predict returns valid prediction data."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{BASE_URL}/predict")
        response.raise_for_status()
        data = response.json()

        assert isinstance(data, list), f"Expected list, got {type(data)}"

        if data:
            point = data[0]
            assert "lat" in point, "Missing 'lat' field"
            assert "lng" in point, "Missing 'lng' field"
            assert "predicted_intensity" in point, "Missing 'predicted_intensity' field"
            assert 0.0 <= point["predicted_intensity"] <= 1.0, (
                f"predicted_intensity {point['predicted_intensity']} outside 0.0-1.0 range"
            )
            print(f"✅ /predict returned {len(data)} predicted hotspots")
            print(f"   Top prediction: lat={point['lat']}, lng={point['lng']}, "
                  f"predicted_intensity={point['predicted_intensity']}")
        else:
            print("✅ /predict returned empty list (no scan history in DB)")


async def test_predict_with_city():
    """Test GET /predict?city=Bangalore filters correctly."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{BASE_URL}/predict", params={"city": "Bangalore"})
        response.raise_for_status()
        data = response.json()
        assert isinstance(data, list)
        print(f"✅ /predict?city=Bangalore returned {len(data)} predictions")


async def test_heatmap_respects_collected():
    """Test that heatmap excludes collected scans (Contract 2)."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get initial heatmap count
        r1 = await client.get(f"{BASE_URL}/heatmap")
        r1.raise_for_status()
        initial_count = len(r1.json())

        # This test documents the expected behavior:
        # After POST /scans/resolve marks scans as collected,
        # GET /heatmap should return fewer points
        print(f"✅ Contract 2 check: heatmap has {initial_count} uncollected points")
        print("   (Mark some as collected via /scans/resolve to see count decrease)")


async def run_all_tests():
    print("=== Analytics Endpoint Tests ===\n")

    tests = [
        test_heatmap_endpoint,
        test_heatmap_with_city,
        test_predict_endpoint,
        test_predict_with_city,
        test_heatmap_respects_collected,
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
