import { useEffect, useState } from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet'
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

const CITY_COORDS = {
  Bangalore: [12.9716, 77.5946],
  Mumbai: [19.0760, 72.8777],
  Delhi: [28.7041, 77.1025],
  Chennai: [13.0827, 80.2707],
}

function getCircleStyle(intensity) {
  if (intensity > 0.7) {
    return { color: '#ef4444', fillColor: '#ef4444', radius: 14 }
  } else if (intensity >= 0.4) {
    return { color: '#f97316', fillColor: '#f97316', radius: 10 }
  } else {
    return { color: '#22c55e', fillColor: '#22c55e', radius: 7 }
  }
}

function MapUpdater({ center }) {
  const map = useMap()
  useEffect(() => {
    map.setView(center, map.getZoom(), { animate: true })
  }, [center, map])
  return null
}

export default function HeatMap({ refreshTrigger, city, children }) {
  const [points, setPoints] = useState([])
  const [lastUpdated, setLastUpdated] = useState(null)
  const [error, setError] = useState(null)

  const center = CITY_COORDS[city] || CITY_COORDS['Bangalore']

  async function fetchHeatmap() {
    try {
      const data = await getHeatmap(city)
      setPoints(Array.isArray(data) ? data : [])
      setLastUpdated(new Date().toLocaleTimeString())
      setError(null)
    } catch (err) {
      console.error('Heatmap fetch failed:', err)
      setError('Could not load heatmap data.')
    }
  }

  useEffect(() => {
    fetchHeatmap()
    const interval = setInterval(fetchHeatmap, 60000)
    return () => clearInterval(interval)
  }, [city])

  useEffect(() => {
    if (refreshTrigger !== undefined && refreshTrigger !== null) {
      fetchHeatmap()
    }
  }, [refreshTrigger])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {error && (
        <div className="eco-status eco-status--error" style={{ borderRadius: 0 }}>
          ⚠️ {error}
        </div>
      )}

      <div style={{ flex: 1, minHeight: 0 }}>
        <MapContainer
          center={center}
          zoom={12}
          style={{ height: '100%', width: '100%' }}
        >
          <MapUpdater center={center} />
          
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {points.map((point, i) => {
            const { color, fillColor, radius } = getCircleStyle(point.intensity)
            return (
              <CircleMarker
                key={`hm-${i}`}
                center={[point.lat, point.lng]}
                radius={radius}
                pathOptions={{
                  color,
                  fillColor,
                  fillOpacity: 0.7,
                  weight: 2,
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

          {children}
        </MapContainer>
      </div>

      <div className="eco-legend">
        <div className="eco-legend__items">
          <span><span className="eco-legend__dot eco-legend__dot--high" /> High</span>
          <span><span className="eco-legend__dot eco-legend__dot--medium" /> Medium</span>
          <span><span className="eco-legend__dot eco-legend__dot--low" /> Low</span>
          <span><span className="eco-legend__dot eco-legend__dot--prediction" /> Predicted</span>
        </div>
        <span>
          {lastUpdated ? `Updated: ${lastUpdated}` : 'Loading...'}
          {' · '}{points.length} hotspots in {city}
        </span>
      </div>
    </div>
  )
}