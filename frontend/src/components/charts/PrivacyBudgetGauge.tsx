import React from 'react';
import { Shield } from 'lucide-react';

interface PrivacyBudgetGaugeProps {
  currentEpsilon: number;
  maxEpsilon?: number;
  delta?: number;
}

export const PrivacyBudgetGauge: React.FC<PrivacyBudgetGaugeProps> = ({
  currentEpsilon,
  maxEpsilon = 5.0,
  delta = 1e-5
}) => {
  const percentage = Math.min(100, Math.max(0, (currentEpsilon / maxEpsilon) * 100));

  let color = 'from-green-500 to-emerald-400';
  let statusText = 'Strict Privacy Guarantee';
  if (percentage > 50 && percentage <= 80) {
    color = 'from-yellow-500 to-amber-400';
    statusText = 'Moderate Privacy Depletion';
  } else if (percentage > 80) {
    color = 'from-red-500 to-rose-400';
    statusText = 'Privacy Threshold Warning';
  }

  return (
    <div className="bg-surface border border-surfaceHighlight rounded-xl p-5 shadow-glow flex flex-col justify-between">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-textMuted flex items-center">
          <Shield className="w-4 h-4 mr-2 text-accent" />
          RDP Privacy Budget Consumption
        </span>
        <span className="text-xs font-mono text-textMuted">δ = {delta.toExponential(1)}</span>
      </div>

      <div className="my-4">
        <div className="flex items-baseline justify-between mb-2">
          <div className="flex items-baseline space-x-1">
            <span className="text-3xl font-bold font-mono text-white">{currentEpsilon.toFixed(2)}</span>
            <span className="text-sm text-textMuted font-mono">/ {maxEpsilon.toFixed(2)} ε</span>
          </div>
          <span className="text-sm font-bold text-accent">{percentage.toFixed(0)}%</span>
        </div>

        <div className="w-full bg-background rounded-full h-3 overflow-hidden border border-white/5">
          <div
            className={`h-full rounded-full bg-gradient-to-r ${color} transition-all duration-700 ease-out`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>

      <div className="text-xs text-textMuted flex items-center justify-between">
        <span>Status: <strong className="text-textMain">{statusText}</strong></span>
        <span className="font-mono">Renyi DP Engine</span>
      </div>
    </div>
  );
};

export default PrivacyBudgetGauge;
