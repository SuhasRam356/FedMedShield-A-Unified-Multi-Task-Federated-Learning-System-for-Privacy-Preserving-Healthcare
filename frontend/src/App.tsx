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
        <div className="bg-amber-950/70 border-b border-amber-500/40 px-6 py-2 text-[11px] text-amber-200 flex items-center justify-between shrink-0 font-medium">
          <div className="flex items-center space-x-2">
            <span className="font-bold tracking-wider text-amber-300">DEMO / SYNTHETIC / NOT FOR CLINICAL OR SECURITY DECISIONS:</span>
            <span>All clinical predictions, drug scores, network alerts, and privacy telemetry are simulated demonstration values. Not for diagnostic, pharmacological, or network defense decisions.</span>
          </div>
          <span className="text-[10px] font-mono text-amber-300/80 ml-4 hidden lg:inline shrink-0">DEMO PROTOTYPE</span>
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
