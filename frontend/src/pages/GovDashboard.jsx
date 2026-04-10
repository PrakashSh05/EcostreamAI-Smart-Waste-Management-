import { useEffect, useState } from 'react'
import HeatMap from '../components/HeatMap'
import PredictiveLayer from '../components/PredictiveLayer'
import { getScans, resolveZone } from '../api/client'

const CITIES = ['Bangalore', 'Mumbai', 'Delhi', 'Chennai']

export default function GovDashboard() {
  const [scans, setScans] = useState([])
  const [selectedCity, setSelectedCity] = useState('Bangalore')
  const [showPredictions, setShowPredictions] = useState(false)
  const [resolveStatus, setResolveStatus] = useState(null)
  const [resolving, setResolving] = useState(false)
  const [refreshTrigger, setRefreshTrigger] = useState(0)
  const [lastUpdated, setLastUpdated] = useState(null)

  useEffect(() => {
    async function fetchScans() {
      try {
        const data = await getScans(selectedCity)
        setScans(data?.items || data || [])
        setLastUpdated(new Date().toLocaleTimeString())
      } catch (err) {
        console.error('Failed to fetch scans:', err)
      }
    }
    fetchScans()
  }, [selectedCity])

  const scansList = Array.isArray(scans) ? scans : []
  const totalScans = scansList.length

  function getTopMaterials() {
    const freq = {}
    scansList.forEach((scan) => {
      (scan.materials || []).forEach((mat) => {
        freq[mat] = (freq[mat] || 0) + 1
      })
    })
    return Object.entries(freq)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([mat, count]) => ({ mat, count }))
  }

  const topMaterials = getTopMaterials()

  async function handleResolve() {
    setResolving(true)
    setResolveStatus(null)
    try {
      const result = await resolveZone(selectedCity, 1.0)
      const resolvedCount = result?.resolved ?? result?.resolved_count ?? 0
      setResolveStatus({
        type: 'success',
        message: `✅ ${resolvedCount} scans marked as collected in ${selectedCity}`,
      })
      setRefreshTrigger((prev) => prev + 1)
    } catch (err) {
      setResolveStatus({
        type: 'error',
        message: '⚠️ Failed to mark as collected. Try again.',
      })
      console.error(err)
    } finally {
      setResolving(false)
    }
  }

  return (
    <div className="eco-dashboard">
      {/* Map Area */}
      <div className="eco-dashboard__map">
        <HeatMap refreshTrigger={refreshTrigger} city={selectedCity}>
          <PredictiveLayer show={showPredictions} city={selectedCity} />
        </HeatMap>
      </div>

      {/* Sidebar */}
      <div className="eco-dashboard__sidebar">
        {/* Sidebar Header */}
        <div>
          <h2 style={{ fontSize: 18, fontWeight: 800, color: 'var(--text-primary)' }}>
            🗺️ Command Center
          </h2>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
            Real-time waste management dashboard
          </p>
        </div>

        {/* Total Scans Stat */}
        <div className="eco-card">
          <div className="eco-stat">
            <div className="eco-stat__icon eco-stat__icon--green">📊</div>
            <div className="eco-stat__info">
              <div className="eco-stat__label">Active Scans</div>
              <div className="eco-stat__value">{totalScans}</div>
            </div>
          </div>
          {lastUpdated && (
            <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
              Last synced: {lastUpdated}
            </p>
          )}
        </div>

        {/* Top Materials */}
        <div className="eco-card">
          <div className="eco-card__title">Top Detected Materials</div>
          {topMaterials.length === 0 ? (
            <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>No data yet</p>
          ) : (
            <div className="eco-material-list">
              {topMaterials.map(({ mat, count }, i) => (
                <div className="eco-material-item" key={mat}>
                  <span className="eco-material-item__name">
                    <span className="eco-material-item__rank">{i + 1}</span>
                    {mat}
                  </span>
                  <span className="eco-material-item__count">{count}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Mark as Collected — Contract 2 */}
        <div className="eco-card">
          <div className="eco-card__title">Zone Collection Control</div>

          <select
            value={selectedCity}
            onChange={(e) => {
              setSelectedCity(e.target.value)
              setResolveStatus(null)
            }}
            className="eco-select"
            style={{ marginBottom: 12 }}
          >
            {CITIES.map((city) => (
              <option key={city} value={city}>{city}</option>
            ))}
          </select>

          <button
            onClick={handleResolve}
            disabled={resolving}
            className={`eco-btn eco-btn--primary eco-btn--full`}
          >
            {resolving ? '⏳ Resolving...' : '✅ Mark Zone as Collected'}
          </button>

          {resolveStatus && (
            <div
              className={`eco-status eco-status--${resolveStatus.type}`}
              style={{ marginTop: 10 }}
            >
              {resolveStatus.message}
            </div>
          )}

          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
            Collected zones are removed from the heatmap immediately
          </p>
        </div>

        {/* Prediction Overlay Toggle */}
        <div className="eco-card">
          <div className="eco-card__title">Prediction Overlay</div>
          <label className="eco-toggle">
            <input
              type="checkbox"
              checked={showPredictions}
              onChange={(e) => setShowPredictions(e.target.checked)}
            />
            <span className="eco-toggle__track" />
            <span className="eco-toggle__label">Show Tomorrow's Hotspots</span>
          </label>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
            <span className="eco-legend__dot eco-legend__dot--prediction" style={{ display: 'inline-block', marginRight: 4, verticalAlign: 'middle' }} />
            Purple markers show predicted waste accumulation points
          </p>
        </div>
      </div>
    </div>
  )
}