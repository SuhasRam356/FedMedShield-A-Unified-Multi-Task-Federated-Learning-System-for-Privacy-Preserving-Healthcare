import React from 'react';
import { Database, Plus, RefreshCw, Signal } from 'lucide-react';
import { useHospitalNodes } from '../hooks/useHospitalNodes';
import HospitalStatusCard from '../components/cards/HospitalStatusCard';
import HospitalRadarChart from '../components/charts/HospitalRadarChart';

export const HospitalNodes: React.FC = () => {
  const { nodes, isLoading } = useHospitalNodes();

  return (
    <div className="space-y-6 fade-in pb-20">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">Hospital Nodes</h1>
          <p className="text-textMuted mt-1">
            Real-time status, hardware capacity, and dataset metrics for federated clinical institutions.
          </p>
        </div>
        <button className="flex items-center space-x-2 bg-accent hover:bg-accentHover text-white px-4 py-2 rounded-lg font-medium shadow-lg transition-all">
          <Plus size={16} />
          <span>Provision Hospital Node</span>
        </button>
      </div>

      {/* Grid of Hospital Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        {nodes.map((node) => (
          <HospitalStatusCard key={node.client_id} node={node} />
        ))}
      </div>

      {/* Bottom Section: Topology & Performance Radar */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-surface border border-surfaceHighlight rounded-xl p-6 shadow-glow">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-lg font-semibold text-white">Federated Institutional Performance</h2>
              <p className="text-sm text-textMuted">Multi-axis comparative evaluation across clinical silos</p>
            </div>
            <span className="text-xs font-mono text-green-400 bg-green-500/10 px-2.5 py-1 rounded-full border border-green-500/20">
              Zero Raw Data Shared
            </span>
          </div>
          <HospitalRadarChart />
        </div>

        <div className="bg-surface border border-surfaceHighlight rounded-xl p-6 shadow-glow flex flex-col justify-between">
          <div>
            <h2 className="text-lg font-semibold text-white mb-2">Hospital Node SLA</h2>
            <p className="text-sm text-textMuted mb-4">
              All 4 medical centers are compliant with HIPAA Title II and GDPR Article 28 data sovereignty agreements.
            </p>

            <div className="space-y-3">
              <div className="p-3 bg-background rounded-lg border border-surfaceHighlight flex justify-between items-center text-xs">
                <span className="text-textMuted">Average Silo Latency</span>
                <span className="font-mono font-bold text-white">28 ms</span>
              </div>
              <div className="p-3 bg-background rounded-lg border border-surfaceHighlight flex justify-between items-center text-xs">
                <span className="text-textMuted">Cumulative Dataset</span>
                <span className="font-mono font-bold text-white">248.5 GB</span>
              </div>
              <div className="p-3 bg-background rounded-lg border border-surfaceHighlight flex justify-between items-center text-xs">
                <span className="text-textMuted">SecAgg Dropout Margin</span>
                <span className="font-mono font-bold text-green-400">1 of 4 Allowed</span>
              </div>
            </div>
          </div>

          <div className="pt-4 border-t border-surfaceHighlight">
            <button className="w-full py-2 bg-background border border-surfaceHighlight rounded-lg text-xs font-medium text-textMuted hover:text-textMain hover:border-accent transition-colors">
              Download Node Compliance Audit (PDF)
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HospitalNodes;
