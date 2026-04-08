import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import CitizenPortal from './pages/CitizenPortal'
import GovDashboard from './pages/GovDashboard'

export default function App() {
  return (
    <BrowserRouter>
      <nav style={{ padding: '10px 20px', borderBottom: '1px solid #ddd', display: 'flex', gap: '20px' }}>
        <Link to="/citizen">🗑️ Citizen Portal</Link>
        <Link to="/dashboard">🗺️ Gov Dashboard</Link>
      </nav>
      <Routes>
        <Route path="/citizen" element={<CitizenPortal />} />
        <Route path="/dashboard" element={<GovDashboard />} />
        <Route path="/" element={<CitizenPortal />} />
      </Routes>
    </BrowserRouter>
  )
}