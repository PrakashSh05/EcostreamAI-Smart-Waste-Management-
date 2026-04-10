import { useEffect, useState } from 'react'
import { CircleMarker, Popup } from 'react-leaflet'
import { getPredictions } from '../api/client'

export default function PredictiveLayer({ show, city }) {
  const [predictions, setPredictions] = useState([])

  useEffect(() => {
    if (!show) return

    async function fetchPredictions() {
      try {
        const data = await getPredictions(city)
        setPredictions(Array.isArray(data) ? data : [])
      } catch (err) {
        console.error('Predictions fetch failed:', err)
      }
    }

    fetchPredictions()
  }, [show, city])

  if (!show) return null

  return (
    <>
      {predictions.map((point, i) => (
        <CircleMarker
          key={`pred-${i}`}
          center={[point.lat, point.lng]}
          radius={point.predicted_intensity * 15}
          pathOptions={{
            color: '#8b5cf6',
            fillColor: '#8b5cf6',
            fillOpacity: 0.3,
            weight: 2,
            dashArray: '4 4',
          }}
        >
          <Popup>
            <div style={{ fontSize: 13, lineHeight: 1.6 }}>
              <strong>🔮 Predicted Hotspot</strong><br />
              Tomorrow's intensity: {(point.predicted_intensity * 100).toFixed(0)}%<br />
              Lat: {point.lat.toFixed(4)}<br />
              Lng: {point.lng.toFixed(4)}
            </div>
          </Popup>
        </CircleMarker>
      ))}
    </>
  )
}
