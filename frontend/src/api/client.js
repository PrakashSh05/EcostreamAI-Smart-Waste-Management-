import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
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

export async function getHeatmap(city) {
  const res = await api.get('/heatmap', { params: { city } })
  return res.data
}

export async function getPredictions(city) {
  const res = await api.get('/predict', { params: { city } })
  return res.data
}

export async function resolveZone(city, radius_km = 1.0) {
  const res = await api.post('/scans/resolve', { city, radius_km })
  return res.data
}

export async function chatWithRAG(message, city = 'Bangalore', imageFile = null) {
  const form = new FormData()
  form.append('message', message)
  form.append('city', city)
  if (imageFile) {
    form.append('file', imageFile)
  }
  const res = await api.post('/chat', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 30000,
  })
  return res.data
}