import React, { useState } from 'react';
import { UploadCloud, Image as ImageIcon, Eye } from 'lucide-react';

interface ImageUploadFormProps {
  onAnalyze: (modality: string, targetCondition: string, patientId: string) => void;
  isLoading?: boolean;
}

export const ImageUploadForm: React.FC<ImageUploadFormProps> = ({ onAnalyze, isLoading = false }) => {
  const [modality, setModality] = useState<string>('MRI');
  const [targetCondition, setTargetCondition] = useState<string>('Brain Tumor Detection');
  const [patientId, setPatientId] = useState<string>('IMG-9941');
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFileName(e.target.files[0].name);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onAnalyze(modality, targetCondition, patientId);
  };

  return (
    <form onSubmit={handleSubmit} className="bg-surface border border-surfaceHighlight rounded-xl p-6 shadow-glow space-y-6">
      <div className="flex items-center space-x-3 border-b border-surfaceHighlight pb-4">
        <Eye className="w-5 h-5 text-accent" />
        <div>
          <h3 className="font-semibold text-textMain text-base">Radiological Scan Submission (Demonstration Pass)</h3>
          <p className="text-xs text-textMuted">Select scan modality and parameters for demonstration analysis pass</p>
        </div>
      </div>


      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-xs font-medium text-textMuted mb-1.5">Imaging Modality</label>
          <select
            value={modality}
            onChange={(e) => setModality(e.target.value)}
            className="w-full bg-background border border-surfaceHighlight rounded-lg px-3 py-2 text-sm text-textMain focus:outline-none focus:border-accent"
          >
            <option value="MRI">Brain MRI (T1/T2-weighted)</option>
            <option value="CT">Computed Tomography (CT)</option>
            <option value="Fundus">Ophthalmic Fundus Photography</option>
            <option value="X-Ray">Chest Radiograph (X-Ray)</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-textMuted mb-1.5">Target Clinical Condition</label>
          <select
            value={targetCondition}
            onChange={(e) => setTargetCondition(e.target.value)}
            className="w-full bg-background border border-surfaceHighlight rounded-lg px-3 py-2 text-sm text-textMain focus:outline-none focus:border-accent"
          >
            <option value="Brain Tumor Detection">Brain Tumor (Glioma/Meningioma)</option>
            <option value="Glaucoma Screening">Glaucoma Optic Neuropathy</option>
            <option value="Pneumonia Infiltration">Pneumonia & Pulmonary Lesions</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-textMuted mb-1.5">Patient / Scan ID</label>
          <input
            type="text"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            className="w-full bg-background border border-surfaceHighlight rounded-lg px-3 py-2 text-sm text-textMain focus:outline-none focus:border-accent font-mono"
            required
          />
        </div>
      </div>

      {/* Drag & Drop Box */}
      <div className="border-2 border-dashed border-surfaceHighlight rounded-xl p-8 text-center hover:border-accent transition-colors cursor-pointer bg-background/50">
        <input
          type="file"
          id="scan-upload"
          className="hidden"
          accept=".dcm,.png,.jpg,.jpeg,.nii"
          onChange={handleFileChange}
        />
        <label htmlFor="scan-upload" className="cursor-pointer flex flex-col items-center">
          <UploadCloud className="w-10 h-10 text-accent mb-2" />
          <p className="text-sm font-medium text-textMain">
            {selectedFileName ? selectedFileName : 'Drag & drop medical DICOM/PNG or click to browse'}
          </p>
          <p className="text-xs text-textMuted mt-1">Supports MRI, CT, Fundus JPG, and NIfTI formats</p>
        </label>
      </div>

      <div className="flex justify-end pt-2">
        <button
          type="submit"
          disabled={isLoading}
          className="px-6 py-2.5 bg-gradient-to-r from-accent to-primary text-white text-sm font-semibold rounded-lg shadow-lg hover:shadow-accent/30 transition-all disabled:opacity-50"
        >
          {isLoading ? 'Analyzing Neural Heatmap...' : 'Execute ResNet18 Classifier'}
        </button>
      </div>
    </form>
  );
};

export default ImageUploadForm;
