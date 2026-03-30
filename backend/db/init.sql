-- backend/db/init.sql

-- Enable extensions for radius-based distance calculations
CREATE EXTENSION IF NOT EXISTS cube;
CREATE EXTENSION IF NOT EXISTS earthdistance;

CREATE TABLE IF NOT EXISTS scans (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    materials TEXT[],
    advice TEXT,
    city VARCHAR(100) DEFAULT 'Bangalore',
    yolo_ms INT,
    rag_ms INT,
    db_ms INT,
    total_ms INT,
    is_collected BOOLEAN DEFAULT FALSE
);

-- Standard indices for searching
CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans (created_at);
CREATE INDEX IF NOT EXISTS idx_scans_city ON scans (city);

-- PERFORMANCE WIN: Partial Index for Member 5's Heatmap
-- This only indexes trash that hasn't been picked up yet.
CREATE INDEX IF NOT EXISTS idx_active_hotspots 
ON scans (city, latitude, longitude) 
WHERE is_collected = FALSE;