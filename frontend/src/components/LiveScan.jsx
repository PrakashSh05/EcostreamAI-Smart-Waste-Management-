import { useRef, useEffect, useState } from 'react'
import imageCompression from 'browser-image-compression'
import { analyzeWaste } from '../api/client'

// Contract 1 — default fallback coords if GPS denied
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
    // Start camera — prefer rear camera on mobile (environment)
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

    // Get GPS — silently fall back to Bangalore if denied
    navigator.geolocation.getCurrentPosition(
      (pos) =>
        setLocation({
          lat: pos.coords.latitude,
          lng: pos.coords.longitude,
        }),
      () => {} // silent fallback to DEFAULT_LAT/LNG
    )

    // Cleanup — stop camera tracks when component unmounts
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
      }
    }
  }, [])

  async function handleScan() {
    if (!videoRef.current || !canvasRef.current) return

    // Reset previous result and errors
    setError(null)
    setResult(null)

    const video = videoRef.current
    const canvas = canvasRef.current

    // Step 1 — capture current video frame onto hidden canvas
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)

    // Step 2 — convert canvas to blob then compress + upload
    canvas.toBlob(async (rawBlob) => {
      try {
        setCompressing(true)
        setLoading(true)

        // Contract 3 — compress BEFORE sending
        // maxSizeMB: 0.3 = 300KB max, maxWidthOrHeight: 640 matches YOLOv11 input size
        const compressed = await imageCompression(rawBlob, {
          maxSizeMB: 0.3,
          maxWidthOrHeight: 640,
        })

        setCompressing(false)

        // Step 3 — send to backend via client.js (never call axios directly here)
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
      {/* Live camera preview */}
      <div
        style={{
          borderRadius: 12,
          overflow: 'hidden',
          background: '#000',
          aspectRatio: '4/3',
          position: 'relative',
        }}
      >
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            display: 'block',
          }}
        />
        {/* Overlay shown before camera stream starts */}
        {!cameraReady && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#fff',
              fontSize: 14,
            }}
          >
            Starting camera...
          </div>
        )}
      </div>

      {/* Hidden canvas — only used to capture video frame */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Scan button — disabled while camera not ready or compressing */}
      <button
        onClick={handleScan}
        disabled={!cameraReady || compressing}
        style={{
          marginTop: 12,
          width: '100%',
          padding: '14px',
          fontSize: 16,
          fontWeight: 600,
          borderRadius: 10,
          border: 'none',
          background: cameraReady && !compressing ? '#16a34a' : '#9ca3af',
          color: '#fff',
          cursor: cameraReady && !compressing ? 'pointer' : 'not-allowed',
          transition: 'background 0.2s',
        }}
      >
        {compressing ? '⏳ Compressing image...' : '📸 Scan Waste'}
      </button>
    </div>
  )
}