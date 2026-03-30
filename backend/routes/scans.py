# backend/routes/scans.py
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor
from backend.db.postgres import db_connection

router = APIRouter(tags=["scans"])

class ResolveScansRequest(BaseModel):
    city: str
    latitude: float | None = None
    longitude: float | None = None
    radius_km: float = 1.0  # Default to 1km radius

@router.get("/scans")
def get_scans(city: str | None = Query(default=None), limit: int | None = Query(default=100, gt=0)):
    """
    Returns UNCOLLECTED scans for Member 5's heatmap.
    """
    with db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT * FROM scans 
                WHERE is_collected = FALSE 
            """
            params = []
            if city:
                query += " AND city = %s"
                params.append(city)
            
            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)

            cur.execute(query, params)
            rows = cur.fetchall()
            return {"count": len(rows), "items": rows}

@router.post("/scans/resolve")
def resolve_scans(payload: ResolveScansRequest):
    """
    Marks scans as collected. If lat/lng provided, clears a radius.
    Otherwise, clears the whole city.
    """
    with db_connection() as conn:
        with conn.cursor() as cur:
            try:
                if payload.latitude and payload.longitude:
                    # Uses earth_distance (result in meters, so we use radius_km * 1000)
                    cur.execute("""
                        UPDATE scans
                        SET is_collected = TRUE
                        WHERE is_collected = FALSE 
                          AND city = %s
                          AND earth_distance(
                              ll_to_earth(latitude, longitude), 
                              ll_to_earth(%s, %s)
                          ) <= %s
                    """, (payload.city, payload.latitude, payload.longitude, payload.radius_km * 1000))
                else:
                    # Clear entire city
                    cur.execute("""
                        UPDATE scans
                        SET is_collected = TRUE
                        WHERE is_collected = FALSE AND city = %s
                    """, (payload.city,))
                
                updated = cur.rowcount
                conn.commit()
                return {"city": payload.city, "resolved": updated, "method": "radius" if payload.latitude else "city-wide"}
            
            except Exception as e:
                conn.rollback()
                raise HTTPException(status_code=500, detail=str(e))