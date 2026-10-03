import React, { useState, useEffect } from 'react';
import { Shield, Lock, Activity, ShieldAlert, Cpu } from 'lucide-react';
import StatCard from '../components/cards/StatCard';

const Privacy: React.FC = () => {
  const [epsilon, setEpsilon] = useState(2.5);
  const [delta, setDelta] = useState(1e-5);
  const [noiseMultiplier, setNoiseMultiplier] = useState(0.85);

  // Simulated real-time secagg status
  const [secAggStatus, setSecAggStatus] = useState("Idle");

  useEffect(() => {
    const timer = setInterval(() => {
      const statuses = ["Idle", "Generating Key Pairs", "Computing Shared Secrets", "Encrypting Updates", "Aggregating Ciphertexts"];
      setSecAggStatus(statuses[Math.floor(Math.random() * statuses.length)]);
    }, 4000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="space-y-6 fade-in">
      <div>
        <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-blue-500">
          Privacy & Secure Aggregation
        </h1>
        <p className="text-textMuted mt-1">
          Monitor Differential Privacy (DP) budgets and Cryptographic Secure Aggregation (SecAgg).
        </p>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatCard
          title="Current ε (Epsilon)"
          value={epsilon.toFixed(2)}
          subtitle="Privacy budget consumed"
          icon={Shield}
        />
        <StatCard
          title="Current δ (Delta)"
          value={delta.toExponential(1)}
          subtitle="Target threshold"
          icon={Activity}
        />
        <StatCard
          title="SecAgg Phase"
          value={secAggStatus}
          subtitle="Cryptographic state"
          icon={Lock}
        />
      </div>

      {/* Main Control Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* DP Configuration */}
        <div className="bg-surface border border-surfaceHighlight rounded-xl p-6 shadow-glow">
          <div className="flex items-center space-x-3 mb-6">
            <ShieldAlert className="text-yellow-400 w-6 h-6" />
            <h2 className="text-xl font-semibold">Differential Privacy Engine</h2>
          </div>
          
          <div className="space-y-6">
            <div>
              <div className="flex justify-between mb-2">
                <label className="text-sm font-medium text-textMuted">Noise Multiplier (σ)</label>
                <span className="text-sm font-bold text-accent">{noiseMultiplier.toFixed(2)}</span>
              </div>
              <input 
                type="range" 
                min="0.1" max="5.0" step="0.05"
                value={noiseMultiplier}
                onChange={(e) => setNoiseMultiplier(parseFloat(e.target.value))}
                className="w-full h-2 bg-background rounded-lg appearance-none cursor-pointer accent-accent"
              />
              <p className="text-xs text-textMuted mt-2">
                Higher noise guarantees stronger privacy but reduces model utility.
              </p>
            </div>

            <div className="p-4 bg-background rounded-lg border border-surfaceHighlight flex items-start space-x-4">
              <Activity className="w-5 h-5 text-accent mt-0.5" />
              <div>
                <h4 className="text-sm font-medium">Gaussian Noise Mechanism</h4>
                <p className="text-xs text-textMuted mt-1">
                  Currently applying calibrated Gaussian noise to model weight updates before transmission to the central server.
                </p>
              </div>
            </div>
            
            <button className="w-full py-2.5 rounded-lg font-semibold bg-gradient-to-r from-accent to-accentHover text-white shadow-lg transition-transform hover:scale-[1.02]">
              Apply DP Parameters
            </button>
          </div>
        </div>

        {/* SecAgg Configuration */}
        <div className="bg-surface border border-surfaceHighlight rounded-xl p-6 shadow-glow">
          <div className="flex items-center space-x-3 mb-6">
            <Cpu className="text-purple-400 w-6 h-6" />
            <h2 className="text-xl font-semibold">Secure Aggregation Topology</h2>
          </div>
          
          <div className="space-y-4">
            <div className="flex justify-between items-center p-3 bg-background rounded border border-surfaceHighlight">
              <div>
                <h4 className="text-sm font-medium">Protocol</h4>
                <p className="text-xs text-textMuted">Bonawitz et al. (Masking)</p>
              </div>
              <span className="px-2 py-1 bg-green-500/20 text-green-400 text-xs font-bold rounded">Active</span>
            </div>

            <div className="flex justify-between items-center p-3 bg-background rounded border border-surfaceHighlight">
              <div>
                <h4 className="text-sm font-medium">Key Exchange</h4>
                <p className="text-xs text-textMuted">Elliptic Curve Diffie-Hellman (ECDH)</p>
              </div>
              <span className="px-2 py-1 bg-green-500/20 text-green-400 text-xs font-bold rounded">Active</span>
            </div>

            <div className="flex justify-between items-center p-3 bg-background rounded border border-surfaceHighlight">
              <div>
                <h4 className="text-sm font-medium">Dropout Tolerance</h4>
                <p className="text-xs text-textMuted">Shamir's Secret Sharing Threshold</p>
              </div>
              <span className="text-sm font-bold text-textMain">3 / 4 Nodes</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default Privacy;
