import { useEffect, useState } from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { getHeatmap } from '../api/client'

// Fix Leaflet marker icon bug with Vite bundler
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

// Per guide — intensity thresholds and colours
function getCircleStyle(intensity) {
  if (intensity > 0.7) {
    return { color: '#ef4444', radius: 14 }   // red — collect urgently
  } else if (intensity >= 0.4) {
    return { color: '#f97316', radius: 10 }   // orange — medium
  } else {
    return { color: '#22c55e', radius: 7 }    // green — low
  }
}

// refreshTrigger — when this value changes, map re-fetches immediately
// children — allows PredictiveLayer to render inside same MapContainer
export default function HeatMap({ refreshTrigger, children }) {
  const [points, setPoints] = useState([])
  const [lastUpdated, setLastUpdated] = useState(null)
  const [error, setError] = useState(null)

  async function fetchHeatmap() {
    try {
      const data = await getHeatmap()
      setPoints(data)
      setLastUpdated(new Date().toLocaleTimeString())
      setError(null)
    } catch (err) {
      console.error('Heatmap fetch failed:', err)
      setError('Could not load heatmap data.')
    }
  }

  // Fetch on mount + set up 60s polling
  useEffect(() => {
    fetchHeatmap()

    const interval = setInterval(() => {
      fetchHeatmap()
    }, 60000) // 60 seconds per guide

    // Cleanup interval when component unmounts
    return () => clearInterval(interval)
  }, [])

  // Re-fetch immediately when refreshTrigger changes
  // This is called by GovDashboard after Mark as Collected succeeds
  useEffect(() => {
    if (refreshTrigger !== undefined && refreshTrigger !== null) {
      fetchHeatmap()
    }
  }, [refreshTrigger])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>

      {/* Error banner */}
      {error && (
        <div
          style={{
            padding: '8px 12px',
            background: '#fef2f2',
            color: '#dc2626',
            fontSize: 13,
            borderBottom: '1px solid #fecaca',
          }}
        >
          ⚠️ {error}
        </div>
      )}

      {/* Leaflet map — MUST have explicit height or it won't render */}
      <div style={{ flex: 1, minHeight: 0 }}>
        <MapContainer
          center={[12.9716, 77.5946]}
          zoom={12}
          style={{ height: '100%', width: '100%' }}
        >
          {/* OpenStreetMap base tiles */}
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {/* Intensity circles — one per heatmap point */}
          {points.map((point, i) => {
            const { color, radius } = getCircleStyle(point.intensity)
            return (
              <CircleMarker
                key={i}
                center={[point.lat, point.lng]}
                radius={radius}
                pathOptions={{
                  color,
                  fillColor: color,
                  fillOpacity: 0.7,
                  weight: 1,
                }}
              >
                <Popup>
                  <div style={{ fontSize: 13, lineHeight: 1.6 }}>
                    <strong>Waste Hotspot</strong><br />
                    Intensity: {(point.intensity * 100).toFixed(0)}%<br />
                    Lat: {point.lat.toFixed(4)}<br />
                    Lng: {point.lng.toFixed(4)}
                  </div>
                </Popup>
              </CircleMarker>
            )
          })}

          {/* PredictiveLayer renders here as a child inside same MapContainer */}
          {children}

        </MapContainer>
      </div>

      {/* Last updated timestamp — visible below map */}
      <div
        style={{
          padding: '6px 12px',
          background: '#f9fafb',
          borderTop: '1px solid #e5e7eb',
          fontSize: 12,
          color: '#9ca3af',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <span>🔴 High &nbsp; 🟠 Medium &nbsp; 🟢 Low</span>
        <span>
          {lastUpdated ? `Updated: ${lastUpdated}` : 'Loading...'}
        </span>
      </div>

    </div>
  )
}