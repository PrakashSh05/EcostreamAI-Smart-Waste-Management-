// Material → chip CSS class mapping
const MATERIAL_CLASS = {
  plastic: 'eco-chip--plastic',
  metal: 'eco-chip--metal',
  food: 'eco-chip--food',
  food_waste: 'eco-chip--food',
  paper: 'eco-chip--paper',
  glass: 'eco-chip--glass',
  cardboard: 'eco-chip--cardboard',
}

function getChipClass(material) {
  const key = Object.keys(MATERIAL_CLASS).find((k) =>
    material.toLowerCase().includes(k)
  )
  return MATERIAL_CLASS[key] || 'eco-chip--default'
}

export default function ScanResult({ result }) {
  if (!result) return null

  const {
    detected_materials = [],
    timing_ms,
  } = result

  return (
    <div className="eco-card eco-advice" style={{ marginTop: 16 }}>
      {/* Detected Materials */}
      <div className="eco-advice__section-title">
        Detected Materials
      </div>

      <div className="eco-chips" style={{ marginBottom: 20 }}>
        {detected_materials.length === 0 ? (
          <span style={{ fontSize: 14, color: 'var(--text-muted)' }}>
            No materials detected — try pointing closer to the waste
          </span>
        ) : (
          detected_materials.map((mat) => (
            <span key={mat} className={`eco-chip ${getChipClass(mat)}`}>
              {mat}
            </span>
          ))
        )}
      </div>

      {/* Success notification instead of disposal advice */}
      {detected_materials.length > 0 && (
        <>
          <div className="eco-advice__section-title">
            Status
          </div>
          <div className="eco-status eco-status--success" style={{ lineHeight: 1.8 }}>
            <span style={{ fontSize: 18, marginRight: 8 }}>✅</span>
            Your waste report has been submitted to the <strong>Government Dashboard</strong>. 
            Municipal authorities have been notified and will take action to collect the waste 
            from your area soon. Thank you for keeping the city clean!
          </div>
        </>
      )}

      {detected_materials.length === 0 && (
        <div className="eco-status eco-status--error" style={{ lineHeight: 1.8 }}>
          <span style={{ fontSize: 18, marginRight: 8 }}>📷</span>
          No waste materials were detected. Try moving closer to the waste or adjusting the camera angle.
        </div>
      )}

      {/* Response time */}
      {timing_ms?.total_ms && (
        <div className="eco-advice__timing">
          ⚡ Response: {(timing_ms.total_ms / 1000).toFixed(1)}s
          {timing_ms.yolo_ms && ` · YOLO: ${timing_ms.yolo_ms}ms`}
        </div>
      )}
    </div>
  )
}