import React from 'react';
import { AlertCircle, CheckCircle2, ShieldAlert, Activity } from 'lucide-react';

interface PredictionResultCardProps {
  patientId: string;
  sepsisScore: number;
  sepsisCategory: string;
  covidProb: number;
  covidSeverity: string;
  confidence: number;
  interventions: string[];
  modelVersion: string;
}

export const PredictionResultCard: React.FC<PredictionResultCardProps> = ({
  patientId,
  sepsisScore,
  sepsisCategory,
  covidProb,
  covidSeverity,
  confidence,
  interventions,
  modelVersion
}) => {
  const isHighRisk = sepsisScore > 0.6 || covidProb > 0.6;

  return (
    <div className="bg-surface border border-surfaceHighlight rounded-xl p-6 shadow-glow space-y-5">
      <div className="flex items-center justify-between border-b border-surfaceHighlight pb-4">
        <div>
          <span className="text-xs font-mono text-textMuted uppercase tracking-wider">Inference Output</span>
          <h3 className="text-xl font-bold text-textMain mt-0.5">Patient Record: {patientId}</h3>
        </div>
        <span className="text-xs font-mono px-3 py-1 rounded bg-background border border-surfaceHighlight text-textMuted">
          {modelVersion}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Sepsis Box */}
        <div className={`p-4 rounded-xl border ${sepsisScore > 0.6 ? 'bg-red-500/10 border-red-500/20' : 'bg-background border-surfaceHighlight'}`}>
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-textMuted">Sepsis 6h Onset Risk</span>
            <span className={`text-xs font-bold uppercase px-2 py-0.5 rounded ${sepsisScore > 0.6 ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
              {sepsisCategory}
            </span>
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-3xl font-bold font-mono text-white">{(sepsisScore * 100).toFixed(1)}%</span>
          </div>
          <p className="text-xs text-textMuted mt-1">SOFA composite clinical indicator</p>
        </div>

        {/* COVID Box */}
        <div className={`p-4 rounded-xl border ${covidProb > 0.6 ? 'bg-yellow-500/10 border-yellow-500/20' : 'bg-background border-surfaceHighlight'}`}>
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-textMuted">COVID-19 Severity Risk</span>
            <span className={`text-xs font-bold uppercase px-2 py-0.5 rounded ${covidProb > 0.6 ? 'bg-yellow-500/20 text-yellow-400' : 'bg-green-500/20 text-green-400'}`}>
              {covidSeverity}
            </span>
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-3xl font-bold font-mono text-white">{(covidProb * 100).toFixed(1)}%</span>
          </div>
          <p className="text-xs text-textMuted mt-1">Multi-modal prognostic biomarker</p>
        </div>
      </div>

      {/* Recommended Clinical Protocol */}
      <div className="p-4 bg-background rounded-xl border border-surfaceHighlight space-y-2">
        <h4 className="text-xs font-semibold text-textMuted uppercase tracking-wider mb-2 flex items-center">
          <Activity className="w-3.5 h-3.5 mr-1.5 text-accent" />
          Educational Simulation Observations (Not Clinical Directives)
        </h4>
        <ul className="space-y-1.5 text-xs text-textMain">
          {interventions.map((action, idx) => (
            <li key={idx} className="flex items-start space-x-2">
              <span className="text-accent font-bold mt-0.5">•</span>
              <span>{action}</span>
            </li>
          ))}
        </ul>
        <div className="mt-2 p-2 rounded bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-300">
          ⚠️ <strong>Research Notice:</strong> Output is a synthetic heuristic score for system prototyping. Do not use for clinical diagnostic or treatment decisions.
        </div>
      </div>

      <div className="text-[11px] text-textMuted flex items-center justify-between pt-2">
        <span>Demonstration Confidence Metric: <strong>{(confidence * 100).toFixed(1)}%</strong></span>
        <span className="text-amber-400/90">Simulated Heuristic Surrogate</span>
      </div>

    </div>
  );
};

export default PredictionResultCard;
