import { useRef, useEffect, useState } from 'react'
import imageCompression from 'browser-image-compression'
import { analyzeWaste } from '../api/client'

const DEFAULT_LAT = 12.9716
const DEFAULT_LNG = 77.5946
const DEFAULT_CITY = 'Bangalore'

export default function LiveScan({ setResult, setLoading, setError }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)

  const [cameraReady, setCameraReady] = useState(false)
  const [compressing, setCompressing] = useState(false)
  const [location, setLocation] = useState({
    lat: DEFAULT_LAT,
    lng: DEFAULT_LNG,
  })

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: 'environment' }, audio: false })
      .then((stream) => {
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
        setCameraReady(true)
      })
      .catch(() =>
        setError('Camera permission denied. Please allow camera access and refresh.')
      )

    navigator.geolocation.getCurrentPosition(
      (pos) =>
        setLocation({
          lat: pos.coords.latitude,
          lng: pos.coords.longitude,
        }),
      () => {}
    )

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
      }
    }
  }, [])

  async function handleScan() {
    if (!videoRef.current || !canvasRef.current) return

    setError(null)
    setResult(null)

    const video = videoRef.current
    const canvas = canvasRef.current

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)

    canvas.toBlob(async (rawBlob) => {
      try {
        setCompressing(true)
        setLoading(true)

        const compressed = await imageCompression(rawBlob, {
          maxSizeMB: 1.0,
          maxWidthOrHeight: 1024,
        })

        setCompressing(false)

        const data = await analyzeWaste(
          compressed,
          location.lat,
          location.lng,
          DEFAULT_CITY
        )

        setResult(data)
      } catch (err) {
        setError('Scan failed. Please try again.')
        console.error(err)
      } finally {
        setLoading(false)
        setCompressing(false)
      }
    }, 'image/jpeg')
  }

  return (
    <div>
      {/* Camera with viewfinder */}
      <div className="eco-camera">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
        />
        {/* Corner viewfinder overlay */}
        <div className="eco-camera__corners" />
        {!cameraReady && (
          <div className="eco-camera__overlay">
            <div className="eco-loader">
              <div className="eco-loader__spinner" />
              <span className="eco-loader__text">Starting camera...</span>
            </div>
          </div>
        )}
      </div>

      {/* Hidden canvas */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Scan button with pulse */}
      <button
        onClick={handleScan}
        disabled={!cameraReady || compressing}
        className={`eco-btn eco-btn--primary eco-btn--full eco-btn--lg ${
          cameraReady && !compressing ? 'eco-btn--scan' : ''
        }`}
        style={{ marginTop: 16 }}
      >
        {compressing ? '⏳ Compressing image...' : '📸 Scan Waste'}
      </button>

      {/* Location info */}
      <p style={{
        textAlign: 'center',
        fontSize: 11,
        color: 'var(--text-muted)',
        marginTop: 8,
      }}>
        📍 {location.lat.toFixed(4)}, {location.lng.toFixed(4)} — {DEFAULT_CITY}
      </p>
    </div>
  )
}