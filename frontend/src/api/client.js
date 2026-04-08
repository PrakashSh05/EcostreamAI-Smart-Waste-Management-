import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
})

export async function analyzeWaste(imageFile, latitude, longitude, city) {
  const form = new FormData()
  form.append('file', imageFile)
  form.append('latitude', latitude)
  form.append('longitude', longitude)
  form.append('city', city)
  const res = await api.post('/analyze', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

export async function getScans(city, limit = 500) {
  const res = await api.get('/scans', { params: { city, limit } })
  return res.data
}

export async function getHeatmap() {
  const res = await api.get('/heatmap')
  return res.data
}

export async function getPredictions() {
  const res = await api.get('/predict')
  return res.data
}

export async function resolveZone(city, radius_km = 1.0) {
  const res = await api.post('/scans/resolve', { city, radius_km })
  return res.data
}