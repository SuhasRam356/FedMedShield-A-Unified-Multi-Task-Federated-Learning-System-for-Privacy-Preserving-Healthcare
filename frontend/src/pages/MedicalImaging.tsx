import React, { useState } from 'react';
import ImageUploadForm from '../components/forms/ImageUploadForm';
import DemoNoticeBanner from '../components/common/DemoNoticeBanner';
import { Eye, ShieldCheck, CheckCircle2, AlertTriangle, Layers } from 'lucide-react';
import imagingService from '../services/imaging.service';

export const MedicalImaging: React.FC = () => {
  const [result, setResult] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAnalyze = async (modality: string, targetCondition: string, patientId: string) => {
    setIsAnalyzing(true);
    try {
      const res = await imagingService.analyzeScanJson({ modality, target_condition: targetCondition, patient_id: patientId });
      setResult(res);
    } catch {
      // High fidelity demo fallback
      setTimeout(() => {
        setResult({
          patient_id: patientId,
          modality: modality,
          target_condition: targetCondition,
          prediction_label: targetCondition.includes('Tumor') ? 'Demonstration Finding: Simulated Lesion Pattern' : 'Demonstration Finding: Simulated Cup Excavation',
          probability: 0.948,
          classification: 'Simulation Demonstration Flagged',
          detected_regions: [
            { label: 'Synthetic Region', confidence: 0.948, box: [0.32, 0.45, 0.68, 0.78] }
          ],
          heatmap_available: true,
          inference_time_ms: 142.5
        });
        setIsAnalyzing(false);
      }, 1200);
      return;
    }
    setIsAnalyzing(false);
  };

  return (
    <div className="space-y-6 fade-in pb-20">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white">Medical Imaging (Module 2: ResNet50)</h1>
        <p className="text-textMuted mt-1">
          Federated radiological deep learning for Brain MRI Glioblastoma detection and Fundus Glaucoma screening.
        </p>
      </div>

      <DemoNoticeBanner
        title="P0 Safety Notice — Synthetic Radiological Simulation"
        message="Radiological inference, bounding boxes, and heatmap overlays displayed here are simulated heuristic placeholders. This demonstration route does not evaluate clinical imaging weights or execute validated diagnostic classification. Do not use for patient triage, radiology diagnosis, or treatment decisions."
        variant="warning"
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        {/* Scan Submission Form */}
        <div>
          <ImageUploadForm onAnalyze={handleAnalyze} isLoading={isAnalyzing} />
        </div>

        {/* Diagnostic Output & Visual Heatmap */}
        <div className="space-y-6">
          {result ? (
            <div className="bg-surface border border-surfaceHighlight rounded-xl p-6 shadow-glow space-y-5">
              <div className="flex items-center justify-between border-b border-surfaceHighlight pb-4">
                <div>
                  <span className="text-xs font-mono text-textMuted uppercase tracking-wider">Demonstration Analysis</span>
                  <h3 className="text-xl font-bold text-textMain mt-0.5">{result.patient_id}</h3>
                </div>
                <span className="text-xs font-mono px-3 py-1 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
                  {result.modality} • {result.inference_time_ms} ms
                </span>
              </div>

              {/* Synthetic Radiological Scan Visualization */}
              <div className="relative aspect-video rounded-xl bg-black border border-surfaceHighlight overflow-hidden flex items-center justify-center">
                {/* Synthetic brain MRI or fundus visual */}
                <div className="w-48 h-48 rounded-full bg-gradient-to-br from-gray-800 via-gray-900 to-black border-4 border-gray-700/50 flex items-center justify-center relative">
                  <div className="w-32 h-32 rounded-full bg-gray-800/80 blur-sm" />
                  {/* Bounding box highlight */}
                  <div className="absolute top-1/4 right-1/4 w-16 h-16 border-2 border-amber-500 rounded bg-amber-500/20 animate-pulse flex items-center justify-center">
                    <span className="text-[9px] font-mono text-amber-300 font-bold bg-black/80 px-1 rounded -top-3 absolute">
                      Simulated Area
                    </span>
                  </div>
                </div>
                <div className="absolute bottom-3 left-3 bg-black/70 backdrop-blur-sm px-2.5 py-1 rounded text-[11px] font-mono text-textMuted border border-white/10">
                  Simulated Localization Overlay (Demo)
                </div>
              </div>

              {/* Classification Results */}
              <div className="p-4 bg-background rounded-xl border border-surfaceHighlight space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-textMuted uppercase font-medium">Demonstration Status</span>
                  <span className="text-xs font-bold text-amber-400 uppercase">{result.classification}</span>
                </div>
                <p className="text-lg font-bold text-white">{result.prediction_label}</p>
                <div className="flex items-center space-x-2 pt-1">
                  <div className="flex-1 bg-surfaceHighlight rounded-full h-2 overflow-hidden">
                    <div className="bg-gradient-to-r from-yellow-500 to-amber-500 h-full rounded-full" style={{ width: `${result.probability * 100}%` }} />
                  </div>
                  <span className="text-xs font-mono font-bold text-white">{(result.probability * 100).toFixed(1)}%</span>
                </div>
                <div className="mt-3 p-2.5 rounded bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-300">
                  ⚠️ <strong>Research Notice:</strong> Output is a demonstration prototype placeholder. Not a clinical radiological diagnosis.
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-surface border border-dashed border-white/10 rounded-xl p-12 text-center flex flex-col items-center justify-center space-y-4 shadow-glow">
              <div className="w-16 h-16 rounded-full bg-accent/10 flex items-center justify-center text-accent">
                <Layers className="w-8 h-8" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-textMain">Awaiting Radiological Scan</h3>
                <p className="text-sm text-textMuted max-w-sm mt-1">
                  Select an imaging modality on the left to run demonstration feature inspection.
                </p>
              </div>
              <div className="flex items-center space-x-2 text-xs text-amber-400 bg-amber-500/10 px-3 py-1 rounded-full border border-amber-500/20">
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>Simulated Imaging Pipeline (Synthetic Display)</span>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
};

export default MedicalImaging;
