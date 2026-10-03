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

import { useAuth } from './hooks/useAuth';

function ProtectedLayout() {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-background text-textMuted text-sm font-mono">
        Validating session credentials...
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return (
    <div className="flex h-screen bg-background text-textMain overflow-hidden font-sans">
      {/* Sidebar Navigation */}
      <Sidebar />

      {/* Main Application Area */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        <Navbar />
        <div className="bg-amber-950/60 border-b border-amber-500/30 px-6 py-2 text-[11px] text-amber-200 flex items-center justify-between shrink-0">
          <div className="flex items-center space-x-2">
            <span className="font-bold uppercase tracking-wider text-amber-300">⚠️ Research Simulation:</span>
            <span>Demonstration prototype only. All clinical predictions, drug scores, network alerts, and privacy telemetry are simulated heuristic values. Not for diagnostic, medical, or clinical decision-making.</span>
          </div>
          <span className="text-[10px] font-mono text-amber-300/80 ml-4 hidden lg:inline shrink-0">DEMO EVALUATION MODE</span>
        </div>
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
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <FLProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/*" element={<ProtectedLayout />} />
          </Routes>
        </BrowserRouter>
      </FLProvider>
    </AuthProvider>
  );
}

export default App;
