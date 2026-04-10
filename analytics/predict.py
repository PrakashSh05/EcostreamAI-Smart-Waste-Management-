"""
analytics/predict.py
Next-day waste hotspot prediction.

Uses time-weighted spatial analysis on historical scan data to predict
where waste will accumulate tomorrow. Recent scans are weighted more
heavily, and day-of-week patterns are factored in.

No external pre-trained model is needed — the scan history in PostgreSQL
IS the training data.
"""

import logging
from datetime import datetime, timezone, timedelta
import numpy as np

logger = logging.getLogger(__name__)

# Day-of-week multipliers (indices 0=Mon ... 6=Sun)
# Waste tends to accumulate more on weekends and after holidays
DAY_WEIGHTS = {
    0: 1.0,   # Monday
    1: 0.9,   # Tuesday
    2: 0.9,   # Wednesday
    3: 1.0,   # Thursday
    4: 1.1,   # Friday
    5: 1.3,   # Saturday
    6: 1.2,   # Sunday
}


def predict_hotspots(scan_rows: list[dict]) -> list[dict]:
    """Predict tomorrow's waste hotspots from historical scan data.

    Parameters
    ----------
    scan_rows : list[dict]
        All scans (both collected and uncollected) from the past 14 days.
        Each dict must have: latitude, longitude, created_at.

    Returns
    -------
    list[dict]
        Each entry: {"lat": float, "lng": float, "predicted_intensity": float (0.0-1.0)}
    """
    if not scan_rows:
        return []

    now = datetime.now(timezone.utc)
    tomorrow = now + timedelta(days=1)
    tomorrow_dow = tomorrow.weekday()  # 0=Mon
    day_weight = DAY_WEIGHTS.get(tomorrow_dow, 1.0)

    # Grid resolution for predictions (~500m cells)
    GRID_RES = 0.005

    # Weighted scoring per grid cell
    cell_scores: dict[tuple[float, float], float] = {}
    cell_counts: dict[tuple[float, float], int] = {}

    for row in scan_rows:
        lat = float(row["latitude"])
        lng = float(row["longitude"])

        # Time decay: more recent scans get higher weight
        created_at = row.get("created_at")
        if created_at:
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    created_at = now
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            days_ago = max(0.1, (now - created_at).total_seconds() / 86400)
        else:
            days_ago = 7.0  # default if no timestamp

        # Exponential time decay: half-life of 3 days
        time_weight = np.exp(-0.231 * days_ago)  # ln(2)/3 ≈ 0.231

        # Same day-of-week bonus: if scan was on same weekday as tomorrow, boost it
        if created_at and hasattr(created_at, "weekday"):
            if created_at.weekday() == tomorrow_dow:
                time_weight *= 1.5

        # Grid cell
        grid_lat = round(float(np.round(lat / GRID_RES) * GRID_RES), 4)
        grid_lng = round(float(np.round(lng / GRID_RES) * GRID_RES), 4)
        key = (grid_lat, grid_lng)

        cell_scores[key] = cell_scores.get(key, 0.0) + time_weight
        cell_counts[key] = cell_counts.get(key, 0) + 1

    if not cell_scores:
        return []

    # Apply day-of-week multiplier and normalize
    max_score = max(cell_scores.values())
    min_score = min(cell_scores.values())
    score_range = max_score - min_score if max_score != min_score else 1.0

    predictions = []
    for (lat, lng), score in cell_scores.items():
        normalized = ((score - min_score) / score_range) * day_weight
        normalized = min(1.0, max(0.1, normalized))  # clamp

        # Only include predictions with meaningful intensity
        if normalized >= 0.15:
            predictions.append({
                "lat": lat,
                "lng": lng,
                "predicted_intensity": round(normalized, 4),
            })

    # Sort by predicted intensity descending
    predictions.sort(key=lambda p: p["predicted_intensity"], reverse=True)

    # Limit to top 50 hotspots to keep response lean
    predictions = predictions[:50]

    logger.info(
        "Prediction: %d historical scans → %d predicted hotspots (tomorrow=%s, day_weight=%.1f)",
        len(scan_rows), len(predictions),
        tomorrow.strftime("%A"), day_weight,
    )

    return predictions
