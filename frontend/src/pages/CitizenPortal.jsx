import { useState } from 'react'
import LiveScan from '../components/LiveScan'
import ChatBot from '../components/ChatBot'
import FloatingChat from '../components/FloatingChat'

export default function CitizenPortal() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  return (
    <div className="eco-page eco-page--center">
      {/* Page Header */}
      <div className="eco-page-header">
        <h2 className="eco-page-header__title">
          <span>📸</span> Scan Your Waste
        </h2>
        <p className="eco-page-header__subtitle">
          Point your camera at waste materials to report it to the government dashboard
        </p>
      </div>

      {/* Camera + Scan button */}
      <LiveScan
        setResult={setResult}
        setLoading={setLoading}
        setError={setError}
      />

      {/* Loading state */}
      {loading && (
        <div className="eco-card" style={{ marginTop: 16 }}>
          <div className="eco-loader">
            <div className="eco-loader__spinner" />
            <span className="eco-loader__text">Analysing your waste...</span>
            <span className="eco-loader__subtext">
              Compressing → Detecting materials → Reporting to dashboard
            </span>
          </div>
        </div>
      )}

      {/* Error state */}
      {error && !loading && (
        <div className="eco-status eco-status--error" style={{ marginTop: 16 }}>
          ⚠️ {error}
        </div>
      )}

      {/* Result — shows government notification instead of disposal advice */}
      {result && !loading && (
        <ChatBot result={result} />
      )}

      {/* Floating AI Assistant Chat */}
      <FloatingChat />
    </div>
  )
}
