import React, { useState } from 'react';
import { Activity, User, Heart, Thermometer, Wind } from 'lucide-react';
import { PatientData } from '../../types/prediction.types';

interface PatientDataFormProps {
  onSubmit: (data: Partial<PatientData>) => void;
  isLoading?: boolean;
}

export const PatientDataForm: React.FC<PatientDataFormProps> = ({ onSubmit, isLoading = false }) => {
  const [formData, setFormData] = useState<Partial<PatientData>>({
    age: 58,
    gender: 'male',
    heartRate: 98,
    bloodPressureSystolic: 115,
    bloodPressureDiastolic: 72,
    respiratoryRate: 22,
    temperature: 38.4,
    oxygenSaturation: 93.5,
    wbc: 14.2,
    creatinine: 1.4,
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value,
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="bg-surface border border-surfaceHighlight rounded-xl p-6 shadow-glow space-y-6">
      <div className="flex items-center space-x-3 border-b border-surfaceHighlight pb-4">
        <Activity className="w-5 h-5 text-accent" />
        <div>
          <h3 className="font-semibold text-textMain text-base">Clinical EHR Vitals Input (Synthetic Simulation)</h3>
          <p className="text-xs text-textMuted">Enter test vitals and laboratory markers for demonstration multi-task scoring</p>
        </div>
      </div>


      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-xs font-medium text-textMuted mb-1.5">Age (Years)</label>
          <input
            type="number"
            name="age"
            value={formData.age}
            onChange={handleChange}
            className="w-full bg-background border border-surfaceHighlight rounded-lg px-3 py-2 text-sm text-textMain focus:outline-none focus:border-accent"
            required
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-textMuted mb-1.5">Gender</label>
          <select
            name="gender"
            value={formData.gender}
            onChange={handleChange}
            className="w-full bg-background border border-surfaceHighlight rounded-lg px-3 py-2 text-sm text-textMain focus:outline-none focus:border-accent"
          >
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-textMuted mb-1.5">Heart Rate (bpm)</label>
          <input
            type="number"
            name="heartRate"
            value={formData.heartRate}
            onChange={handleChange}
            className="w-full bg-background border border-surfaceHighlight rounded-lg px-3 py-2 text-sm text-textMain focus:outline-none focus:border-accent"
            required
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-textMuted mb-1.5">Systolic BP (mmHg)</label>
          <input
            type="number"
            name="bloodPressureSystolic"
            value={formData.bloodPressureSystolic}
            onChange={handleChange}
            className="w-full bg-background border border-surfaceHighlight rounded-lg px-3 py-2 text-sm text-textMain focus:outline-none focus:border-accent"
            required
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-textMuted mb-1.5">Diastolic BP (mmHg)</label>
          <input
            type="number"
            name="bloodPressureDiastolic"
            value={formData.bloodPressureDiastolic}
            onChange={handleChange}
            className="w-full bg-background border border-surfaceHighlight rounded-lg px-3 py-2 text-sm text-textMain focus:outline-none focus:border-accent"
            required
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-textMuted mb-1.5">Respiratory Rate (/min)</label>
          <input
            type="number"
            name="respiratoryRate"
            value={formData.respiratoryRate}
            onChange={handleChange}
            className="w-full bg-background border border-surfaceHighlight rounded-lg px-3 py-2 text-sm text-textMain focus:outline-none focus:border-accent"
            required
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-textMuted mb-1.5">Temperature (°C)</label>
          <input
            type="number"
            step="0.1"
            name="temperature"
            value={formData.temperature}
            onChange={handleChange}
            className="w-full bg-background border border-surfaceHighlight rounded-lg px-3 py-2 text-sm text-textMain focus:outline-none focus:border-accent"
            required
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-textMuted mb-1.5">SpO2 Oxygen (%)</label>
          <input
            type="number"
            step="0.1"
            name="oxygenSaturation"
            value={formData.oxygenSaturation}
            onChange={handleChange}
            className="w-full bg-background border border-surfaceHighlight rounded-lg px-3 py-2 text-sm text-textMain focus:outline-none focus:border-accent"
            required
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-textMuted mb-1.5">WBC Count (k/µL)</label>
          <input
            type="number"
            step="0.1"
            name="wbc"
            value={formData.wbc}
            onChange={handleChange}
            className="w-full bg-background border border-surfaceHighlight rounded-lg px-3 py-2 text-sm text-textMain focus:outline-none focus:border-accent"
            required
          />
        </div>
      </div>

      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2 border-t border-surfaceHighlight/50">
        <p className="text-[11px] text-amber-300/80 leading-tight">
          ⚠️ <strong>Privacy Notice:</strong> Submitting sends these vitals to the API server (<code className="text-amber-200">/api/prediction/ehr</code>). Do not enter real patient data or PHI.
        </p>
        <button
          type="submit"
          disabled={isLoading}
          className="shrink-0 px-6 py-2.5 bg-gradient-to-r from-accent to-primary text-white text-sm font-semibold rounded-lg shadow-lg hover:shadow-accent/30 transition-all disabled:opacity-50"
        >
          {isLoading ? 'Running Inference...' : 'Evaluate Multi-Task Risk'}
        </button>
      </div>
    </form>
  );
};

export default PatientDataForm;
