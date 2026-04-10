import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import CitizenPortal from './pages/CitizenPortal'
import GovDashboard from './pages/GovDashboard'

export default function App() {
  return (
    <BrowserRouter>
      {/* Premium Navbar */}
      <nav className="eco-navbar">
        <div className="eco-navbar__brand">
          <div className="eco-navbar__brand-icon">♻️</div>
          <span className="eco-navbar__brand-text">EcoStream AI</span>
        </div>
        <div className="eco-navbar__links">
          <NavLink
            to="/citizen"
            className={({ isActive }) =>
              `eco-navbar__link ${isActive ? 'eco-navbar__link--active' : ''}`
            }
          >
            <span className="eco-navbar__link-icon">📸</span>
            Citizen Portal
          </NavLink>
          <NavLink
            to="/dashboard"
            className={({ isActive }) =>
              `eco-navbar__link ${isActive ? 'eco-navbar__link--active' : ''}`
            }
          >
            <span className="eco-navbar__link-icon">🗺️</span>
            Gov Dashboard
          </NavLink>
        </div>
      </nav>

      <Routes>
        <Route path="/citizen" element={<CitizenPortal />} />
        <Route path="/dashboard" element={<GovDashboard />} />
        <Route path="/" element={<CitizenPortal />} />
      </Routes>
    </BrowserRouter>
  )
}