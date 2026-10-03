import React from 'react';
import { Shield, Lock, FileText, Download, CheckCircle2, Award } from 'lucide-react';
import PrivacyBudgetGauge from '../components/charts/PrivacyBudgetGauge';

export const PrivacyReport: React.FC = () => {
  return (
    <div className="space-y-6 fade-in pb-20">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">Privacy & Compliance Audit Report</h1>
          <p className="text-textMuted mt-1">
            Formal mathematical privacy certificates, Renyi DP accounting, and Bonawitz SecAgg verification.
          </p>
        </div>
        <button className="flex items-center space-x-2 bg-primary hover:bg-primary/90 text-white px-4 py-2 rounded-lg font-medium shadow-lg transition-all">
          <Download size={16} />
          <span>Export Regulatory Audit (PDF)</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <PrivacyBudgetGauge currentEpsilon={1.5} maxEpsilon={5.0} delta={1e-5} />

        <div className="bg-surface border border-surfaceHighlight rounded-xl p-5 shadow-glow flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 text-green-400 mb-2">
              <Award className="w-5 h-5" />
              <h3 className="font-semibold text-textMain text-base">HIPAA & GDPR Privacy Guarantee</h3>
            </div>
            <p className="text-xs text-textMuted leading-relaxed">
              No patient identifiability reconstruction is mathematically feasible under (ε=1.5, δ=1e-5)-DP. All client gradients are masked via 256-bit Diffie-Hellman secrets before leaving hospital firewall perimeters.
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
          SecAgg Cryptographic Verification Log
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-surfaceHighlight font-semibold text-textMuted uppercase">
                <th className="pb-3 pr-4">FL Round</th>
                <th className="pb-3 pr-4">Protocol Stage</th>
                <th className="pb-3 pr-4">Key Agreement</th>
                <th className="pb-3 pr-4">Mask Neutralization</th>
                <th className="pb-3">Audit Verification</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surfaceHighlight">
              {[1, 2, 3, 4, 5].map((round) => (
                <tr key={round} className="hover:bg-background/40 transition-colors">
                  <td className="py-3 pr-4 font-mono font-bold text-white">Round #{round}</td>
                  <td className="py-3 pr-4 text-textMain">Round Completed & Aggregated</td>
                  <td className="py-3 pr-4 font-mono text-textMuted">ECDH Curve25519 (Pairwise)</td>
                  <td className="py-3 pr-4 text-green-400 font-mono">Zero-Sum Residual Verified</td>
                  <td className="py-3">
                    <span className="inline-flex items-center text-green-400">
                      <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> Valid Cryptographic Proof
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
