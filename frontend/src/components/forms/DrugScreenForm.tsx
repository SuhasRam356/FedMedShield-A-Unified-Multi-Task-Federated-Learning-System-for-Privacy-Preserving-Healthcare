import React, { useState } from 'react';
import { FlaskConical, Atom } from 'lucide-react';
import { validateSMILES } from '../../utils/validators';

interface DrugScreenFormProps {
  onScreen: (compoundId: string, smiles: string, targetProtein: string) => void;
  isLoading?: boolean;
}

export const DrugScreenForm: React.FC<DrugScreenFormProps> = ({ onScreen, isLoading = false }) => {
  const [compoundId, setCompoundId] = useState<string>('CMP-8042');
  const [smiles, setSmiles] = useState<string>('CC(=O)Oc1ccccc1C(=O)O');
  const [targetProtein, setTargetProtein] = useState<string>('EGFR_HUMAN (Kinase Domain)');
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateSMILES(smiles)) {
      setError('Invalid chemical SMILES string syntax.');
      return;
    }
    setError(null);
    onScreen(compoundId, smiles, targetProtein);
  };

  return (
    <form onSubmit={handleSubmit} className="bg-surface border border-surfaceHighlight rounded-xl p-6 shadow-glow space-y-6">
      <div className="flex items-center space-x-3 border-b border-surfaceHighlight pb-4">
        <FlaskConical className="w-5 h-5 text-accent" />
        <div>
          <h3 className="font-semibold text-textMain text-base">Small Molecule Candidate Screening (Demonstration)</h3>
          <p className="text-xs text-textMuted">Evaluate chemical SMILES against target protein via simulated affinity estimation</p>
        </div>
      </div>


      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-xs font-medium text-textMuted mb-1.5">Compound Identifier</label>
          <input
            type="text"
            value={compoundId}
            onChange={(e) => setCompoundId(e.target.value)}
            className="w-full bg-background border border-surfaceHighlight rounded-lg px-3 py-2 text-sm text-textMain focus:outline-none focus:border-accent font-mono"
            required
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-textMuted mb-1.5">Target Biological Protein</label>
          <select
            value={targetProtein}
            onChange={(e) => setTargetProtein(e.target.value)}
            className="w-full bg-background border border-surfaceHighlight rounded-lg px-3 py-2 text-sm text-textMain focus:outline-none focus:border-accent"
          >
            <option value="EGFR_HUMAN (Kinase Domain)">EGFR_HUMAN (Epidermal Growth Factor Receptor)</option>
            <option value="SARS_COV2_Mpro">SARS-CoV-2 Main Protease (Mpro / 3CLpro)</option>
            <option value="HER2_HUMAN (Oncogene)">HER2_HUMAN (Receptor Tyrosine Kinase)</option>
            <option value="ACE2_HUMAN">ACE2_HUMAN (Angiotensin Converting Enzyme 2)</option>
          </select>
        </div>
      </div>

      <div>
        <label className="block text-xs font-medium text-textMuted mb-1.5">Chemical Structure (Canonical SMILES)</label>
        <div className="relative">
          <input
            type="text"
            value={smiles}
            onChange={(e) => {
              setSmiles(e.target.value);
              setError(null);
            }}
            placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
            className="w-full bg-background border border-surfaceHighlight rounded-lg pl-9 pr-4 py-2 text-sm text-textMain focus:outline-none focus:border-accent font-mono"
            required
          />
          <Atom className="w-4 h-4 text-textMuted absolute left-3 top-1/2 -translate-y-1/2" />
        </div>
        {error && <p className="text-xs text-red-400 mt-1">{error}</p>}
      </div>

      <div className="flex justify-end pt-2">
        <button
          type="submit"
          disabled={isLoading}
          className="px-6 py-2.5 bg-gradient-to-r from-accent to-primary text-white text-sm font-semibold rounded-lg shadow-lg hover:shadow-accent/30 transition-all disabled:opacity-50"
        >
          {isLoading ? 'Computing Binding Affinity...' : 'Screen Bioactivity & ADMET'}
        </button>
      </div>
    </form>
  );
};

export default DrugScreenForm;
