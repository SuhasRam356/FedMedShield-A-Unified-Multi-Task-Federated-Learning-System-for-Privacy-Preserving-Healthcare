import React, { useState } from 'react';
import { Play, Settings2 } from 'lucide-react';

interface TaskFormProps {
  onStart: (taskType: string, rounds: number, dpEpsilon: number) => void;
  isStarting: boolean;
}

const TaskForm: React.FC<TaskFormProps> = ({ onStart, isStarting }) => {
  const [taskType, setTaskType] = useState('ehr');
  const [rounds, setRounds] = useState(10);
  const [epsilon, setEpsilon] = useState(5.0);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onStart(taskType, rounds, epsilon);
  };

  return (
    <form onSubmit={handleSubmit} className="glass-panel p-5 rounded-xl border border-white/10 w-full max-w-sm">
      <div className="flex items-center space-x-2 mb-4">
        <Settings2 size={18} className="text-primary" />
        <h3 className="text-white font-medium">Configure FL Task</h3>
      </div>
      
      <div className="space-y-4">
        <div>
          <label className="block text-xs text-textMuted mb-1.5 uppercase tracking-wide">AI Module</label>
          <select 
            value={taskType} 
            onChange={(e) => setTaskType(e.target.value)}
            className="w-full bg-background border border-white/10 rounded-lg p-2 text-sm text-textMain outline-none focus:border-primary transition-colors"
          >
            <option value="ehr">EHR Risk Prediction</option>
            <option value="imaging_tumor">Tumor Detection</option>
            <option value="imaging_glaucoma">Glaucoma Detection</option>
            <option value="drug">Drug Binding Affinity</option>
            <option value="ids">Network IDS</option>
          </select>
        </div>

        <div className="flex space-x-4">
          <div className="flex-1">
            <label className="block text-xs text-textMuted mb-1.5 uppercase tracking-wide">Target Rounds</label>
            <input 
              type="number" 
              min="1" 
              max="100"
              value={rounds}
              onChange={(e) => setRounds(parseInt(e.target.value))}
              className="w-full bg-background border border-white/10 rounded-lg p-2 text-sm text-textMain outline-none focus:border-primary transition-colors"
            />
          </div>
          <div className="flex-1">
            <label className="block text-xs text-textMuted mb-1.5 uppercase tracking-wide">DP Target (ε)</label>
            <input 
              type="number" 
              step="0.1"
              min="0.1" 
              max="20.0"
              value={epsilon}
              onChange={(e) => setEpsilon(parseFloat(e.target.value))}
              className="w-full bg-background border border-white/10 rounded-lg p-2 text-sm text-textMain outline-none focus:border-primary transition-colors"
            />
          </div>
        </div>

        <button 
          type="submit"
          disabled={isStarting}
          className="w-full mt-2 flex items-center justify-center space-x-2 bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-white p-2.5 rounded-lg transition-colors font-medium shadow-lg shadow-primary/20 disabled:opacity-50"
        >
          {isStarting ? (
             <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
          ) : (
            <>
              <Play size={16} fill="currentColor" />
              <span>Launch Orchestration</span>
            </>
          )}
        </button>
      </div>
    </form>
  );
};

export default TaskForm;
