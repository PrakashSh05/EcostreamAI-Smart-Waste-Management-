import { useEffect, useState } from 'react'
import HeatMap from '../components/HeatMap'
import PredictiveLayer from '../components/PredictiveLayer'
import { getScans, resolveZone } from '../api/client'

const CITIES = ['Bangalore', 'Mumbai', 'Delhi', 'Chennai']

export default function GovDashboard() {
  const [scans, setScans] = useState([])
  const [selectedCity, setSelectedCity] = useState('Bangalore')
  const [showPredictions, setShowPredictions] = useState(false)
  const [resolveStatus, setResolveStatus] = useState(null) // success/error message
  const [resolving, setResolving] = useState(false)
  const [refreshTrigger, setRefreshTrigger] = useState(0)
  const [lastUpdated, setLastUpdated] = useState(null)

  // Fetch scan logs for sidebar stats on mount
  useEffect(() => {
    async function fetchScans() {
      try {
        const data = await getScans()
        setScans(data)
        setLastUpdated(new Date().toLocaleTimeString())
      } catch (err) {
        console.error('Failed to fetch scans:', err)
      }
    }
    fetchScans()
  }, [])

  // Calculate total scans count
  const totalScans = scans.length

  // Calculate top 3 materials by frequency across all scans
  function getTopMaterials() {
    const freq = {}
    scans.forEach((scan) => {
      (scan.materials || []).forEach((mat) => {
        freq[mat] = (freq[mat] || 0) + 1
      })
    })
    return Object.entries(freq)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([mat, count]) => ({ mat, count }))
  }

  const topMaterials = getTopMaterials()

  // Contract 2 — Mark as Collected handler
  async function handleResolve() {
    setResolving(true)
    setResolveStatus(null)

    try {
      const result = await resolveZone(selectedCity, 1.0)

      // Show success message with count of resolved rows
      setResolveStatus({
        type: 'success',
        message: `✅ ${result.resolved_count ?? result} scans marked as collected in ${selectedCity}`,
      })

      // Increment refreshTrigger → HeatMap re-fetches immediately
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
    <div style={{ display: 'flex', height: 'calc(100vh - 41px)' }}>

      {/* LEFT — Map area (70% width) */}
      <div style={{ flex: 7, position: 'relative' }}>
        <HeatMap refreshTrigger={refreshTrigger}>
          {/* PredictiveLayer renders inside HeatMap's MapContainer */}
          <PredictiveLayer show={showPredictions} />
        </HeatMap>
      </div>

      {/* RIGHT — Sidebar (30% width) */}
      <div
        style={{
          flex: 3,
          background: '#fff',
          borderLeft: '1px solid #e5e7eb',
          padding: 20,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 20,
        }}
      >
        {/* Sidebar header */}
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#1a1a18' }}>
            🗺️ Government Dashboard
          </h2>
          <p style={{ fontSize: 12, color: '#9ca3af', marginTop: 4 }}>
            Bengaluru Waste Management
          </p>
        </div>

        {/* Stats card — total scans */}
        <div
          style={{
            background: '#f9fafb',
            borderRadius: 10,
            padding: 14,
            border: '1px solid #e5e7eb',
          }}
        >
          <p style={{ fontSize: 12, color: '#9ca3af', marginBottom: 4 }}>
            Total Scans
          </p>
          <p style={{ fontSize: 28, fontWeight: 700, color: '#1a1a18' }}>
            {totalScans}
          </p>
          {lastUpdated && (
            <p style={{ fontSize: 11, color: '#9ca3af', marginTop: 4 }}>
              Last updated: {lastUpdated}
            </p>
          )}
        </div>

        {/* Top 3 materials */}
        <div
          style={{
            background: '#f9fafb',
            borderRadius: 10,
            padding: 14,
            border: '1px solid #e5e7eb',
          }}
        >
          <p style={{ fontSize: 12, color: '#9ca3af', marginBottom: 10 }}>
            Top Detected Materials
          </p>
          {topMaterials.length === 0 ? (
            <p style={{ fontSize: 13, color: '#9ca3af' }}>No data yet</p>
          ) : (
            topMaterials.map(({ mat, count }, i) => (
              <div
                key={mat}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: 8,
                }}
              >
                <span style={{ fontSize: 13, color: '#374151' }}>
                  {i + 1}. {mat}
                </span>
                <span
                  style={{
                    fontSize: 12,
                    background: '#e0e7ff',
                    color: '#3730a3',
                    padding: '2px 8px',
                    borderRadius: 10,
                    fontWeight: 600,
                  }}
                >
                  {count}
                </span>
              </div>
            ))
          )}
        </div>

        {/* Contract 2 — City dropdown + Mark as Collected */}
        <div
          style={{
            background: '#f9fafb',
            borderRadius: 10,
            padding: 14,
            border: '1px solid #e5e7eb',
          }}
        >
          <p style={{ fontSize: 12, color: '#9ca3af', marginBottom: 8 }}>
            Mark Zone as Collected
          </p>

          {/* City dropdown */}
          <select
            value={selectedCity}
            onChange={(e) => {
              setSelectedCity(e.target.value)
              setResolveStatus(null)
            }}
            style={{
              width: '100%',
              padding: '8px 10px',
              borderRadius: 8,
              border: '1px solid #d1d5db',
              fontSize: 14,
              marginBottom: 10,
              background: '#fff',
              color: '#1a1a18',
            }}
          >
            {CITIES.map((city) => (
              <option key={city} value={city}>
                {city}
              </option>
            ))}
          </select>

          {/* Mark as Collected button — Contract 2 */}
          <button
            onClick={handleResolve}
            disabled={resolving}
            style={{
              width: '100%',
              padding: '10px',
              fontSize: 14,
              fontWeight: 600,
              borderRadius: 8,
              border: 'none',
              background: resolving ? '#9ca3af' : '#2563eb',
              color: '#fff',
              cursor: resolving ? 'not-allowed' : 'pointer',
            }}
          >
            {resolving ? '⏳ Resolving...' : '✅ Mark as Collected'}
          </button>

          {/* Success or error message after resolve */}
          {resolveStatus && (
            <div
              style={{
                marginTop: 10,
                padding: '8px 10px',
                borderRadius: 8,
                fontSize: 13,
                background: resolveStatus.type === 'success' ? '#f0fdf4' : '#fef2f2',
                color: resolveStatus.type === 'success' ? '#166534' : '#dc2626',
                border: `1px solid ${resolveStatus.type === 'success' ? '#bbf7d0' : '#fecaca'}`,
              }}
            >
              {resolveStatus.message}
            </div>
          )}
        </div>

        {/* Predictive layer toggle */}
        <div
          style={{
            background: '#f9fafb',
            borderRadius: 10,
            padding: 14,
            border: '1px solid #e5e7eb',
          }}
        >
          <p style={{ fontSize: 12, color: '#9ca3af', marginBottom: 10 }}>
            Overlay
          </p>
          <label
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              cursor: 'pointer',
              fontSize: 14,
              color: '#374151',
            }}
          >
            <input
              type="checkbox"
              checked={showPredictions}
              onChange={(e) => setShowPredictions(e.target.checked)}
              style={{ width: 16, height: 16, cursor: 'pointer' }}
            />
            Show Predicted Hotspots
            <span
              style={{
                width: 12,
                height: 12,
                borderRadius: '50%',
                background: '#8b5cf6',
                display: 'inline-block',
              }}
            />
          </label>
          <p style={{ fontSize: 11, color: '#9ca3af', marginTop: 6 }}>
            Purple markers show tomorrow's predicted waste hotspots
          </p>
        </div>

      </div>
    </div>
  )
}

                         