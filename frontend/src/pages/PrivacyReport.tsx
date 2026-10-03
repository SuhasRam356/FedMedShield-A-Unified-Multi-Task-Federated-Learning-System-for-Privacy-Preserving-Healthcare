import React from 'react';
import DemoNoticeBanner from '../components/common/DemoNoticeBanner';
import { Shield, Lock, FileText, Download, CheckCircle2, Award } from 'lucide-react';
import PrivacyBudgetGauge from '../components/charts/PrivacyBudgetGauge';

export const PrivacyReport: React.FC = () => {
  return (
    <div className="space-y-6 fade-in pb-20">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">Privacy & Compliance Telemetry (Simulation)</h1>
          <p className="text-textMuted mt-1">
            Demonstration of DP-SGD budget accounting concepts and pairwise masking workflow telemetry.
          </p>
        </div>
        <button className="flex items-center space-x-2 bg-primary/80 hover:bg-primary text-white px-4 py-2 rounded-lg font-medium shadow-lg transition-all">
          <Download size={16} />
          <span>Export Demonstration Log (PDF)</span>
        </button>
      </div>

      <DemoNoticeBanner
        title="P0 Privacy Telemetry Notice — Simulated Privacy Budget & Verification Display"
        message="The differential privacy budgets (ε, δ), clipping bounds, and secure aggregation logs presented below are simulated demonstration metrics. They do not represent a formally proven or calibrated patient-level DP-SGD guarantee, nor do they substantiate legal HIPAA or GDPR compliance certifications. Do not use for regulatory or compliance audits."
        variant="warning"
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <PrivacyBudgetGauge currentEpsilon={1.5} maxEpsilon={5.0} delta={1e-5} />

        <div className="bg-surface border border-surfaceHighlight rounded-xl p-5 shadow-glow flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 text-amber-400 mb-2">
              <Award className="w-5 h-5" />
              <h3 className="font-semibold text-textMain text-base">Demonstration Privacy Profile (Simulated DP-SGD)</h3>
            </div>
            <p className="text-xs text-textMuted leading-relaxed">
              Nominal simulation configuration targeting (ε=1.5, δ=1e-5). Formal patient-level differential privacy in production requires calibrated subsampling, empirical membership-inference auditing, and certified cryptographic key-agreement before regulatory compliance certification.
            </p>
          </div>

          <div className="mt-4 p-3 bg-background rounded-lg border border-surfaceHighlight grid grid-cols-2 gap-3 text-xs">
            <div>
              <span className="text-textMuted block">Clipping Bound (C):</span>
              <span className="font-mono font-bold text-white">1.00 L2 Norm</span>
            </div>
            <div>
              <span className="text-textMuted block">Noise Multiplier (σ):</span>
              <span className="font-mono font-bold text-white">0.85 Gaussian</span>
            </div>
          </div>
        </div>
      </div>

      {/* Cryptographic Verification Table */}
      <div className="bg-surface border border-surfaceHighlight rounded-xl p-6 shadow-glow">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Lock className="w-5 h-5 mr-2 text-accent" />
          Demonstration SecAgg Protocol Workflow Log
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-surfaceHighlight font-semibold text-textMuted uppercase">
                <th className="pb-3 pr-4">FL Round</th>
                <th className="pb-3 pr-4">Protocol Stage</th>
                <th className="pb-3 pr-4">Key Agreement</th>
                <th className="pb-3 pr-4">Mask Neutralization</th>
                <th className="pb-3">Verification Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surfaceHighlight">
              {[1, 2, 3, 4, 5].map((round) => (
                <tr key={round} className="hover:bg-background/40 transition-colors">
                  <td className="py-3 pr-4 font-mono font-bold text-white">Round #{round}</td>
                  <td className="py-3 pr-4 text-textMain">Round Completed & Aggregated</td>
                  <td className="py-3 pr-4 font-mono text-textMuted">Pairwise Masking (Prototype)</td>
                  <td className="py-3 pr-4 text-amber-400 font-mono">Zero-Sum Residual Verified (Simulated)</td>
                  <td className="py-3">
                    <span className="inline-flex items-center text-amber-400">
                      <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> Simulated Verification Check
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
};

export default PrivacyReport;
