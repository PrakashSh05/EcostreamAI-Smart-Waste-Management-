import { useState } from 'react'
import LiveScan from '../components/LiveScan'
import ChatBot from '../components/ChatBot'

export default function CitizenPortal() {
  // These 3 states are passed down to LiveScan as setters
  // LiveScan calls them during the scan flow
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  return (
    <div
      style={{
        maxWidth: 480,
        margin: '0 auto',
        padding: 16,
        minHeight: '100vh',
      }}
    >
      {/* Page header */}
      <h2
        style={{
          fontSize: 20,
          fontWeight: 700,
          marginBottom: 16,
          color: '#1a1a18',
        }}
      >
        ♻️ Scan Your Waste
      </h2>

      {/* Camera + scan button */}
      <LiveScan
        setResult={setResult}
        setLoading={setLoading}
        setError={setError}
      />

      {/* Loading state — shown while backend is processing */}
      {loading && (
        <div
          style={{
            marginTop: 16,
            padding: 20,
            textAlign: 'center',
            background: '#fff',
            borderRadius: 12,
            border: '1px solid #e5e7eb',
          }}
        >
          <div
            style={{
              fontSize: 28,
              marginBottom: 8,
              animation: 'spin 1s linear infinite',
            }}
          >
            🔍
          </div>
          <p style={{ color: '#6b7280', fontSize: 14 }}>
            Analysing your waste...
          </p>
          <p style={{ color: '#9ca3af', fontSize: 12, marginTop: 4 }}>
            Compressing image → Detecting materials → Getting advice
          </p>
        </div>
      )}

      {/* Error state — shown if scan fails */}
      {error && !loading && (
        <div
          style={{
            marginTop: 16,
            padding: 14,
            background: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: 10,
            color: '#dc2626',
            fontSize: 14,
          }}
        >
          ⚠️ {error}
        </div>
      )}

      {/* Result — shown when backend responds successfully */}
      {result && !loading && (
        <ChatBot result={result} />
      )}

      {/* Spin animation for the loading icon */}
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}
