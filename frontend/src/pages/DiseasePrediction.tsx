import React, { useState } from 'react';
import PatientDataForm from '../components/forms/PatientDataForm';
import PredictionResultCard from '../components/cards/PredictionResultCard';
import DemoNoticeBanner from '../components/common/DemoNoticeBanner';
import { usePrediction } from '../hooks/usePrediction';
import { Activity, ShieldCheck, FileCheck } from 'lucide-react';

export const DiseasePrediction: React.FC = () => {
  const { result, isPredicting, predict } = usePrediction();
  const [hasEvaluated, setHasEvaluated] = useState(false);

  const handlePredict = async (data: any) => {
    await predict(data);
    setHasEvaluated(true);
  };

  return (
    <div className="space-y-6 fade-in pb-20">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white">Disease Prediction (Module 1: EHR)</h1>
        <p className="text-textMuted mt-1">
          Multi-task clinical prognostic inference for Sepsis 6-hour onset and COVID-19 ICU mortality risk.
        </p>
      </div>

      <DemoNoticeBanner
        title="P0 Safety & Data Flow Notice — Synthetic Clinical Simulation"
        message="Patient vitals entered below are transmitted to the central demonstration API (/api/prediction/ehr) and scored using a demonstration heuristic surrogate. This route does not run validated clinical models and is not on-premise local inference. Do not input real patient data or make clinical decisions based on these values."
        variant="warning"
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        {/* Input Form */}
        <div>
          <PatientDataForm onSubmit={handlePredict} isLoading={isPredicting} />
        </div>

        {/* Inference Results View */}
        <div className="space-y-6">
          {hasEvaluated && result ? (
            <PredictionResultCard
              patientId={result.patient_id}
              sepsisScore={result.sepsis_risk_score}
              sepsisCategory={result.sepsis_risk_category}
              covidProb={result.covid_outcome_prob}
              covidSeverity={result.covid_severity}
              confidence={result.confidence}
              interventions={result.recommended_interventions}
              modelVersion={result.model_version}
            />
          ) : (
            <div className="bg-surface border border-dashed border-white/10 rounded-xl p-12 text-center flex flex-col items-center justify-center space-y-4 shadow-glow">
              <div className="w-16 h-16 rounded-full bg-accent/10 flex items-center justify-center text-accent">
                <Activity className="w-8 h-8" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-textMain">Awaiting Patient EHR Vitals</h3>
                <p className="text-sm text-textMuted max-w-sm mt-1">
                  Submit the patient form on the left to evaluate simulated multi-task prognosis.
                </p>
              </div>
              <div className="flex items-center space-x-2 text-xs text-amber-400 bg-amber-500/10 px-3 py-1 rounded-full border border-amber-500/20">
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>Simulated Research Pipeline (Synthetic Parameters Only)</span>
              </div>
            </div>
          )}


          {/* Model Information Box */}
          <div className="bg-surface border border-surfaceHighlight rounded-xl p-5 shadow-glow space-y-3">
            <h4 className="text-xs font-semibold text-textMuted uppercase tracking-wider flex items-center">
              <FileCheck className="w-4 h-4 mr-2 text-primary" />
              Federated Multi-Task Architecture
            </h4>
            <p className="text-xs text-textMuted leading-relaxed">
              Trained across 4 decentralized hospital silos without raw data pooling. Utilizes shared temporal feature representations with task-specific projection heads for sepsis risk calibration and COVID severity classification.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DiseasePrediction;
