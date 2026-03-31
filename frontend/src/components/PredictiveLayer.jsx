import { useEffect, useState } from 'react'
import { CircleMarker, Popup } from 'react-leaflet'
import { getPredictions } from '../api/client'

// show prop — returns null immediately if toggle is off
export default function PredictiveLayer({ show }) {
  const [predictions, setPredictions] = useState([])
  const [error, setError] = useState(null)

  useEffect(() => {
    // Only fetch if the layer is toggled on
    if (!show) return

    async function fetchPredictions() {
      try {
        const data = await getPredictions()
        setPredictions(data)
        setError(null)
      } catch (err) {
        console.error('Predictions fetch failed:', err)
        setError('Could not load predictions.')
      }
    }

    fetchPredictions()
  }, [show]) // re-fetches every time user toggles the layer on

  // Per guide — return null if show is false
  if (!show) return null

  return (
    <>
      {predictions.map((point, i) => (
        <CircleMarker
          key={i}
          center={[point.lat, point.lng]}
          // Per guide — radius scales with predicted_intensity * 15
          radius={point.predicted_intensity * 15}
          pathOptions={{
            color: '#8b5cf6',           // purple — distinct from live markers
            fillColor: '#8b5cf6',
            fillOpacity: 0.3,           // semi-transparent per guide
            weight: 2,
            dashArray: '4 4',           // dashed border to distinguish from live
          }}
        >
          <Popup>
            <div style={{ fontSize: 13, lineHeight: 1.6 }}>
              <strong>Predicted Hotspot</strong><br />
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


