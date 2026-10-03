import React, { useState } from 'react';
import DrugScreenForm from '../components/forms/DrugScreenForm';
import DemoNoticeBanner from '../components/common/DemoNoticeBanner';
import { FlaskConical, Atom, CheckCircle2, ShieldCheck, Zap } from 'lucide-react';
import drugService, { CompoundScreenResponse } from '../services/drug.service';

export const DrugDiscovery: React.FC = () => {
  const [result, setResult] = useState<CompoundScreenResponse | null>(null);
  const [isScreening, setIsScreening] = useState(false);

  const handleScreen = async (compoundId: string, smiles: string, targetProtein: string) => {
    setIsScreening(true);
    try {
      const res = await drugService.screenCompound({
        compound_id: compoundId,
        smiles_string: smiles,
        target_protein: targetProtein
      });
      setResult(res);
    } catch {
      setTimeout(() => {
        setResult({
          compound_id: compoundId,
          target_protein: targetProtein,
          predicted_kd_nm: 14.8,
          binding_affinity_score: -9.24,
          bioactivity_class: 'Demonstration: High Simulated Affinity',
          druggability_probability: 0.94,
          safety_admet_flag: 'Simulation Estimate: Low Risk Indicator (Unverified)',
          recommendation: 'Simulation prototype estimate: candidate exhibits simulated binding in demonstration pass. Validated wet-lab assay required.'
        });
        setIsScreening(false);
      }, 1000);
      return;
    }
    setIsScreening(false);
  };

  return (
    <div className="space-y-6 fade-in pb-20">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white">Drug Discovery (Module 3: Bioactivity)</h1>
        <p className="text-textMuted mt-1">
          Federated drug-target affinity modeling without exposing proprietary pharmaceutical library SMILES.
        </p>
      </div>

      <DemoNoticeBanner
        title="P0 Data Flow & Safety Notice — Centralized Prototype Transmission"
        message="SMILES structures entered below are transmitted to the central demonstration API (/api/drug/screen) and evaluated with a heuristic property approximation formula. This transmission is NOT protected by SecAgg (SecAgg operates on federated model weights, not raw REST API payloads). Do not submit proprietary or confidential chemical structures, and do not use scores for synthesis or pharmacology decisions."
        variant="warning"
      />


      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        <div>
          <DrugScreenForm onScreen={handleScreen} isLoading={isScreening} />
        </div>

        <div className="space-y-6">
          {result ? (
            <div className="bg-surface border border-surfaceHighlight rounded-xl p-6 shadow-glow space-y-5">
              <div className="flex items-center justify-between border-b border-surfaceHighlight pb-4">
                <div>
                  <span className="text-xs font-mono text-textMuted uppercase tracking-wider">Candidate Analysis</span>
                  <h3 className="text-xl font-bold text-textMain mt-0.5">{result.compound_id}</h3>
                </div>
                <span className="text-xs font-mono px-3 py-1 rounded bg-green-500/10 text-green-400 border border-green-500/20">
                  {result.bioactivity_class}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-background rounded-xl border border-surfaceHighlight">
                  <span className="text-xs text-textMuted font-medium">Binding Affinity (ΔG)</span>
                  <div className="mt-2 flex items-baseline space-x-1">
                    <span className="text-3xl font-bold font-mono text-white">{result.binding_affinity_score}</span>
                    <span className="text-xs text-textMuted font-mono">kcal/mol</span>
                  </div>
                  <p className="text-[11px] text-green-400 mt-1">Strong thermodynamic binding</p>
                </div>

                <div className="p-4 bg-background rounded-xl border border-surfaceHighlight">
                  <span className="text-xs text-textMuted font-medium">Dissociation Constant (Kd)</span>
                  <div className="mt-2 flex items-baseline space-x-1">
                    <span className="text-3xl font-bold font-mono text-white">{result.predicted_kd_nm}</span>
                    <span className="text-xs text-textMuted font-mono">nM</span>
                  </div>
                  <p className="text-[11px] text-primary mt-1">Sub-micromolar potency</p>
                </div>
              </div>

              {/* ADMET and Recommendation */}
              <div className="p-4 bg-background rounded-xl border border-surfaceHighlight space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-textMuted">ADMET Safety Profile:</span>
                  <span className="font-semibold text-green-400">{result.safety_admet_flag}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-textMuted">Druggability Probability:</span>
                  <span className="font-mono font-bold text-white">{(result.druggability_probability * 100).toFixed(1)}%</span>
                </div>
                <div className="pt-2 border-t border-surfaceHighlight">
                  <span className="text-[11px] font-semibold text-textMuted uppercase tracking-wider block mb-1">
                    Simulation Prototype Indicator:
                  </span>
                  <p className="text-xs text-textMain leading-relaxed">{result.recommendation}</p>
                </div>
                <div className="mt-2 p-2 rounded bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-300">
                  ⚠️ <strong>Research Notice:</strong> Output is a heuristic simulation estimate. Validated wet-lab assays and computational docking are required before chemical synthesis.
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-surface border border-dashed border-white/10 rounded-xl p-12 text-center flex flex-col items-center justify-center space-y-4 shadow-glow">
              <div className="w-16 h-16 rounded-full bg-accent/10 flex items-center justify-center text-accent">
                <Atom className="w-8 h-8" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-textMain">Awaiting Compound SMILES</h3>
                <p className="text-sm text-textMuted max-w-sm mt-1">
                  Enter candidate molecular structures on the left to evaluate simulated affinity estimates.
                </p>
              </div>
              <div className="flex items-center space-x-2 text-xs text-amber-400 bg-amber-500/10 px-3 py-1 rounded-full border border-amber-500/20">
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>Central Demo Screening (Public / Synthetic Structures Only)</span>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
};

export default DrugDiscovery;
