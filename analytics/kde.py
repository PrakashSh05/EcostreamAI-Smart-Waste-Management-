"""
analytics/kde.py
Kernel Density Estimation for waste-dump heatmap.

Reads UNCOLLECTED scans from the database and computes a spatial intensity
score for each coordinate using Gaussian KDE (scikit-learn).

Key Contract:
    Only scans where is_collected = FALSE are included.
    When an operator marks a zone as collected via POST /scans/resolve,
    those scans are excluded and the heatmap clears.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_heatmap(scan_rows: list[dict]) -> list[dict]:
    """Compute KDE-based heatmap from uncollected scan rows.

    Parameters
    ----------
    scan_rows : list[dict]
        Each dict must have 'latitude' and 'longitude' keys.
        These should ONLY be uncollected scans (is_collected = FALSE).

    Returns
    -------
    list[dict]
        Each entry: {"lat": float, "lng": float, "intensity": float (0.0-1.0)}
    """
    if not scan_rows:
        return []

    lats = np.array([float(r["latitude"]) for r in scan_rows])
    lngs = np.array([float(r["longitude"]) for r in scan_rows])

    # If only 1 point, return it with 0.35 intensity (Low/Green)
    if len(lats) == 1:
        return [{"lat": float(lats[0]), "lng": float(lngs[0]), "intensity": 0.35}]

    # Grid-based KDE approach: cluster nearby points and compute density
    # Use a grid resolution of ~0.005 degrees (~500m cells)
    GRID_RES = 0.005

    # Round to grid cells
    grid_lats = np.round(lats / GRID_RES) * GRID_RES
    grid_lngs = np.round(lngs / GRID_RES) * GRID_RES

    # Count scans per grid cell
    cell_counts: dict[tuple[float, float], int] = {}
    for glat, glng in zip(grid_lats, grid_lngs):
        key = (round(float(glat), 4), round(float(glng), 4))
        cell_counts[key] = cell_counts.get(key, 0) + 1

    if not cell_counts:
        return []

    # Normalize counts to 0.0 - 1.0
    max_count = max(cell_counts.values())
    min_count = min(cell_counts.values())
    count_range = max_count - min_count if max_count != min_count else 1

    points = []
    for (lat, lng), count in cell_counts.items():
        intensity = (count - min_count) / count_range
        
        # Absolute thresholds (same as sklearn logic)
        if count <= 2:
            intensity = min(0.35, intensity)
        elif count <= 5:
            intensity = min(0.65, max(0.40, intensity))
        else:
            intensity = max(0.75, intensity)

        # Apply floor to ensure even single-scan cells show up
        intensity = max(0.15, intensity)
        points.append({
            "lat": lat,
            "lng": lng,
            "intensity": round(intensity, 4),
        })

    # Sort by intensity descending (hottest spots first)
    points.sort(key=lambda p: p["intensity"], reverse=True)

    logger.info(
        "KDE heatmap: %d scans → %d grid cells, max_count=%d",
        len(scan_rows), len(points), max_count,
    )

    return points


def compute_heatmap_sklearn(scan_rows: list[dict]) -> list[dict]:
    """Advanced KDE using scikit-learn's KernelDensity (if available).

    Falls back to grid-based approach if sklearn is not available.
    """
    try:
        from sklearn.neighbors import KernelDensity
    except ImportError:
        logger.warning("scikit-learn not available, using grid-based KDE")
        return compute_heatmap(scan_rows)

    if not scan_rows or len(scan_rows) < 2:
        return compute_heatmap(scan_rows)

    lats = np.array([float(r["latitude"]) for r in scan_rows])
    lngs = np.array([float(r["longitude"]) for r in scan_rows])

    coords = np.column_stack([lats, lngs])

    # Bandwidth of ~0.005 radians ≈ 500m
    kde = KernelDensity(bandwidth=0.005, kernel="gaussian", metric="haversine")

    # Convert to radians for haversine
    coords_rad = np.radians(coords)
    kde.fit(coords_rad)

    # Score each original point
    log_densities = kde.score_samples(coords_rad)
    densities = np.exp(log_densities)

    # Normalize to 0.0 - 1.0
    d_min, d_max = densities.min(), densities.max()
    if d_max > d_min:
        normalized = (densities - d_min) / (d_max - d_min)
    else:
        normalized = np.ones_like(densities)

    # Cluster into grid cells to avoid returning 1000s of points
    GRID_RES = 0.003
    cell_data: dict[tuple[float, float], list[float]] = {}
    cell_counts: dict[tuple[float, float], int] = {}

    for i in range(len(lats)):
        key = (
            round(float(np.round(lats[i] / GRID_RES) * GRID_RES), 4),
            round(float(np.round(lngs[i] / GRID_RES) * GRID_RES), 4),
        )
        if key not in cell_data:
            cell_data[key] = []
            cell_counts[key] = 0
        cell_data[key].append(float(normalized[i]))
        cell_counts[key] += 1

    points = []
    for (lat, lng), intensities in cell_data.items():
        # Mix relative KDE density with absolute count penalty
        # 1-2 scans -> Cap at Low/Green (< 0.4)
        # 3-4 scans -> Cap at Medium/Orange (< 0.7)
        # 5+ scans -> Cap at High/Red
        raw_count = cell_counts[(lat, lng)]
        
        base_intensity = max(intensities)
        
        # Absolute thresholds
        if raw_count <= 2:
            intensity = min(0.35, base_intensity) # Force green
        elif raw_count <= 5:
            intensity = min(0.65, max(0.40, base_intensity)) # Force orange
        else:
            intensity = max(0.75, base_intensity) # Force red

        intensity = max(0.15, intensity)  # floor visibility
        points.append({
            "lat": lat,
            "lng": lng,
            "intensity": round(intensity, 4),
        })

    points.sort(key=lambda p: p["intensity"], reverse=True)

    logger.info(
        "sklearn KDE heatmap: %d scans → %d cells",
        len(scan_rows), len(points),
    )

    return points
