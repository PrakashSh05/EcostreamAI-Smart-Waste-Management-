// Colour map per guide — Section 4, ChatBot.jsx spec
const MATERIAL_COLORS = {
  plastic: { bg: '#fee2e2', color: '#991b1b' },  // red
  metal:   { bg: '#f3f4f6', color: '#374151' },  // grey
  food:    { bg: '#dcfce7', color: '#166534' },  // green
  paper:   { bg: '#fef9c3', color: '#854d0e' },  // yellow
  default: { bg: '#e0e7ff', color: '#3730a3' },  // indigo
}

// Case-insensitive match — "Dirty Plastic" still matches "plastic"
function getMaterialColor(material) {
  const key = Object.keys(MATERIAL_COLORS).find((k) =>
    material.toLowerCase().includes(k)
  )
  return MATERIAL_COLORS[key ?? 'default']
}

export default function ChatBot({ result }) {
  // Return nothing if no result yet
  if (!result) return null

  const {
    detected_materials = [],
    disposal_advice = '',
    timing_ms,
  } = result

  return (
    <div
      style={{
        marginTop: 16,
        background: '#fff',
        borderRadius: 12,
        border: '1px solid #e5e7eb',
        padding: 16,
      }}
    >
      {/* Section 1 — Material chips */}
      <p style={{ fontSize: 12, color: '#9ca3af', marginBottom: 8 }}>
        Detected Materials
      </p>

      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 6,
          marginBottom: 16,
        }}
      >
        {detected_materials.length === 0 ? (
          <span style={{ fontSize: 14, color: '#9ca3af' }}>
            No materials detected
          </span>
        ) : (
          detected_materials.map((mat) => {
            const { bg, color } = getMaterialColor(mat)
            return (
              <span
                key={mat}
                style={{
                  background: bg,
                  color,
                  padding: '4px 12px',
                  borderRadius: 20,
                  fontSize: 13,
                  fontWeight: 600,
                }}
              >
                {mat}
              </span>
            )
          })
        )}
      </div>

      {/* Section 2 — Disposal advice text */}
      <p style={{ fontSize: 12, color: '#9ca3af', marginBottom: 6 }}>
        Disposal Advice
      </p>
      <p
        style={{
          fontSize: 14,
          lineHeight: 1.7,
          color: '#1a1a18',
        }}
      >
        {disposal_advice}
      </p>

      {/* Section 3 — Response time caption (optional, from timing_ms) */}
      {timing_ms?.total_ms && (
        <p
          style={{
            marginTop: 12,
            fontSize: 11,
            color: '#d1d5db',
            textAlign: 'right',
          }}
        >
          Response time: {(timing_ms.total_ms / 1000).toFixed(1)}s
        </p>
      )}
    </div>
  )
}