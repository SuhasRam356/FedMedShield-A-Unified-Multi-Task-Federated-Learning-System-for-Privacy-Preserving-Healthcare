import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { FLProvider } from './context/FLContext';
import Sidebar from './components/layout/Sidebar';
import Navbar from './components/layout/Navbar';

// Pages
import Dashboard from './pages/Dashboard';
import HospitalNodes from './pages/HospitalNodes';
import DiseasePrediction from './pages/DiseasePrediction';
import MedicalImaging from './pages/MedicalImaging';
import DrugDiscovery from './pages/DrugDiscovery';
import IntrusionDetection from './pages/IntrusionDetection';
import PrivacyReport from './pages/PrivacyReport';
import Settings from './pages/Settings';
import Login from './pages/Login';

function App() {
  return (
    <AuthProvider>
      <FLProvider>
        <BrowserRouter>
          <div className="flex h-screen bg-background text-textMain overflow-hidden font-sans">
            {/* Sidebar Navigation */}
            <Sidebar />

            {/* Main Application Area */}
            <div className="flex-1 flex flex-col h-screen overflow-hidden">
              <Navbar />
              <main className="flex-1 overflow-y-auto p-6 scroll-smooth">
                <Routes>
                  <Route path="/" element={<Navigate to="/dashboard" replace />} />
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/nodes" element={<HospitalNodes />} />
                  <Route path="/prediction" element={<DiseasePrediction />} />
                  <Route path="/imaging" element={<MedicalImaging />} />
                  <Route path="/drug" element={<DrugDiscovery />} />
                  <Route path="/ids" element={<IntrusionDetection />} />
                  <Route path="/privacy" element={<PrivacyReport />} />
                  <Route path="/settings" element={<Settings />} />
                  <Route path="/login" element={<Login />} />
                  <Route path="*" element={<Navigate to="/dashboard" replace />} />
                </Routes>
              </main>
            </div>
          </div>
        </BrowserRouter>
      </FLProvider>
    </AuthProvider>
  );
}

export default App;
