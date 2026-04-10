"""
analytics/routes/predict.py
FastAPI router exposing heatmap and prediction endpoints.

GET /heatmap   — Live waste hotspot density (uncollected scans only)
GET /predict   — Tomorrow's predicted hotspots (all recent scans)

Contract 2 compliance:
    /heatmap only queries scans WHERE is_collected = FALSE.
    After POST /scans/resolve marks scans as collected, they vanish
    from the heatmap on the next refresh.
"""

import logging
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Query
from psycopg2.extras import RealDictCursor

from backend.db.postgres import db_connection
from analytics.kde import compute_heatmap_sklearn
from analytics.predict import predict_hotspots

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analytics"])


@router.get("/heatmap")
def get_heatmap(city: str | None = Query(default=None)):
    """Return KDE heatmap points for UNCOLLECTED scans only.

    Contract 2: is_collected = FALSE filter ensures collected zones
    disappear from the heatmap immediately after resolve.
    """
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT latitude, longitude
                FROM scans
                WHERE is_collected = FALSE
            """
            params = []
            if city:
                query += " AND city = %s"
                params.append(city)

            query += " ORDER BY created_at DESC LIMIT 500"
            cur.execute(query, params)
            rows = cur.fetchall()

    if not rows:
        return []

    return compute_heatmap_sklearn(rows)


@router.get("/predict")
def get_predictions(city: str | None = Query(default=None)):
    """Return predicted hotspots for tomorrow based on 14-day scan history.

    Uses ALL scans (collected + uncollected) from the past 14 days
    because prediction needs the full historical pattern, not just
    what's currently uncollected.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=14)

    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT latitude, longitude, created_at
                FROM scans
                WHERE created_at >= %s
            """
            params: list = [cutoff]
            if city:
                query += " AND city = %s"
                params.append(city)

            query += " ORDER BY created_at DESC LIMIT 1000"
            cur.execute(query, params)
            rows = cur.fetchall()

    if not rows:
        return []

    return predict_hotspots(rows)
