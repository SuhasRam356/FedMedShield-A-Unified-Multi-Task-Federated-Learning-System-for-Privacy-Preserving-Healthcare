import { useState, useEffect, useCallback, useRef } from 'react';
import { Play, Square, Activity, Shield, Cpu, TrendingDown, Zap, Clock, BarChart3 } from 'lucide-react';
import LossChart from '../components/charts/LossChart';

/* ═══════════════════════════════════════════════════════════════
   Simulation Engine — generates realistic FL training curves
   so the dashboard is always alive and demonstrable.
   ═══════════════════════════════════════════════════════════════ */

interface SimulationState {
  isRunning: boolean;
  taskType: string;
  currentRound: number;
  totalRounds: number;
  currentLoss: number;
  currentAccuracy: number;
  dpEpsilon: number;
  activeNodes: number;
  lossHistory: { round: number; loss: number; accuracy: number }[];
  roundLog: { round: number; timestamp: string; loss: number; accuracy: number; nodesReporting: number }[];
  elapsedSeconds: number;
  status: 'idle' | 'initializing' | 'training' | 'aggregating' | 'completed';
}

const INITIAL_STATE: SimulationState = {
  isRunning: false,
  taskType: '',
  currentRound: 0,
  totalRounds: 10,
  currentLoss: 0,
  currentAccuracy: 0,
  dpEpsilon: 0,
  activeNodes: 0,
  lossHistory: [],
  roundLog: [],
  elapsedSeconds: 0,
  status: 'idle',
};

// Realistic loss curves: start high, converge with noise
function generateLoss(round: number, taskType: string): number {
  const base = taskType === 'ehr' ? 2.8 : 3.5;
  const decay = taskType === 'ehr' ? 0.35 : 0.28;
  const noise = (Math.random() - 0.5) * 0.15;
  return Math.max(0.08, base * Math.exp(-decay * round) + noise);
}

function generateAccuracy(round: number, taskType: string): number {
  const ceiling = taskType === 'ehr' ? 0.94 : 0.91;
  const rate = taskType === 'ehr' ? 0.3 : 0.25;
  const noise = (Math.random() - 0.5) * 0.03;
  return Math.min(0.99, ceiling * (1 - Math.exp(-rate * round)) + noise);
}

const Dashboard = () => {
  const [sim, setSim] = useState<SimulationState>(INITIAL_STATE);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const hasAutoStarted = useRef(false);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const startTask = useCallback((taskType: string) => {
    // Clear any previous intervals
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (timerRef.current) clearInterval(timerRef.current);

    const totalRounds = 10;

    setSim({
      isRunning: true,
      taskType,
      currentRound: 0,
      totalRounds,
      currentLoss: taskType === 'ehr' ? 2.85 : 3.52,
      currentAccuracy: 0,
      dpEpsilon: 0,
      activeNodes: 4,
      lossHistory: [],
      roundLog: [],
      elapsedSeconds: 0,
      status: 'initializing',
    });

    // After a brief "initializing" phase, begin training
    setTimeout(() => {
      let round = 0;

      // Elapsed time counter
      timerRef.current = setInterval(() => {
        setSim(prev => ({ ...prev, elapsedSeconds: prev.elapsedSeconds + 1 }));
      }, 1000);

      intervalRef.current = setInterval(() => {
        round += 1;

        if (round > totalRounds) {
          if (intervalRef.current) clearInterval(intervalRef.current);
          if (timerRef.current) clearInterval(timerRef.current);
          setSim(prev => ({ ...prev, isRunning: false, status: 'completed' }));
          return;
        }

        const loss = generateLoss(round, taskType);
        const accuracy = generateAccuracy(round, taskType);
        const nodesReporting = Math.random() > 0.15 ? 4 : 3; // occasional dropout
        const now = new Date();

        setSim(prev => ({
          ...prev,
          currentRound: round,
          currentLoss: loss,
          currentAccuracy: accuracy,
          dpEpsilon: +(round * 0.15).toFixed(2),
          activeNodes: nodesReporting,
          status: round % 2 === 0 ? 'aggregating' : 'training',
          lossHistory: [
            ...prev.lossHistory,
            { round, loss, accuracy },
          ],
          roundLog: [
            { round, timestamp: now.toLocaleTimeString(), loss, accuracy, nodesReporting },
            ...prev.roundLog,
          ].slice(0, 20), // keep last 20
        }));
      }, 2000); // one round every 2 seconds
    }, 1500); // 1.5s initializing delay
  }, []);

  const stopTask = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (timerRef.current) clearInterval(timerRef.current);
    setSim(prev => ({ ...prev, isRunning: false, status: 'idle' }));
  }, []);

  // Auto-start an EHR simulation on mount so the dashboard is always alive
  useEffect(() => {
    if (!hasAutoStarted.current) {
      hasAutoStarted.current = true;
      const t = setTimeout(() => startTask('ehr'), 500);
      return () => clearTimeout(t);
    }
  }, [startTask]);

  const progressPercent = sim.totalRounds > 0 ? (sim.currentRound / sim.totalRounds) * 100 : 0;

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
  };

  const statusColor: Record<string, string> = {
    idle: 'text-gray-400',
    initializing: 'text-yellow-400',
    training: 'text-blue-400',
    aggregating: 'text-purple-400',
    completed: 'text-green-400',
  };

  const statusLabel: Record<string, string> = {
    idle: 'Idle',
    initializing: 'Initializing Nodes...',
    training: 'Local Training',
    aggregating: 'Secure Aggregation',
    completed: 'Training Complete ✓',
  };

  return (
    <div className="space-y-6 pb-20 fade-in">
      {/* Header */}
      <div className="flex justify-between items-end">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-bold tracking-tight text-white">Dashboard</h1>
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
              Simulation & Live Monitor Demo Mode
            </span>
          </div>
          <p className="text-textMuted mt-1">
            Monitor federated learning orchestration, live curves, and simulated multi-task telemetry.
          </p>
        </div>
        
        {sim.isRunning ? (
          <button 
            onClick={stopTask}
            className="flex items-center space-x-2 bg-red-500/20 text-red-400 hover:bg-red-500/30 px-5 py-2.5 rounded-lg transition-colors font-medium border border-red-500/20"
          >
            <Square size={16} />
            <span>Stop Orchestration</span>
          </button>
        ) : (
          <div className="flex space-x-3">
            <button 
              onClick={() => startTask('ehr')}
              className="flex items-center space-x-2 bg-primary hover:bg-primary/90 text-white px-5 py-2.5 rounded-lg transition-all font-medium shadow-lg shadow-primary/20 hover:shadow-primary/40 hover:scale-[1.02]"
            >
              <Play size={16} fill="currentColor" />
              <span>Start EHR Task</span>
            </button>
            <button 
              onClick={() => startTask('imaging_tumor')}
              className="flex items-center space-x-2 bg-card hover:bg-card/80 text-white px-5 py-2.5 rounded-lg transition-all font-medium border border-white/10 hover:border-white/20 hover:scale-[1.02]"
            >
              <Play size={16} fill="currentColor" />
              <span>Start Imaging Task</span>
            </button>
          </div>
        )}
      </div>

      {/* Live Status Banner */}
      {sim.status !== 'idle' && (
        <div className={`flex items-center justify-between px-5 py-3 rounded-lg border ${
          sim.status === 'completed' 
            ? 'bg-green-500/10 border-green-500/20' 
            : 'bg-card/50 border-white/5'
        }`}>
          <div className="flex items-center space-x-3">
            {sim.isRunning && (
              <div className="w-2.5 h-2.5 rounded-full bg-green-400 animate-pulse" />
            )}
            <span className={`font-medium ${statusColor[sim.status]}`}>
              {statusLabel[sim.status]}
            </span>
            <span className="text-textMuted text-sm">
              — {sim.taskType === 'ehr' ? 'EHR Classification' : 'Tumor Imaging'} 
              {sim.taskType === 'ehr' ? ' (Multi-Label)' : ' (Segmentation)'}
            </span>
          </div>
          <div className="flex items-center space-x-2 text-textMuted text-sm">
            <Clock size={14} />
            <span className="font-mono">{formatTime(sim.elapsedSeconds)}</span>
          </div>
        </div>
      )}

      {/* Top Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
        {/* Round Progress */}
        <div className="glass-panel rounded-xl p-5 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Activity size={80} />
          </div>
          <p className="text-sm text-textMuted font-medium">Round Progress</p>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-4xl font-bold text-white">{sim.currentRound}</span>
            <span className="text-sm text-textMuted">/ {sim.totalRounds}</span>
          </div>
          <div className="w-full bg-background rounded-full h-1.5 mt-4 overflow-hidden border border-white/5">
            <div 
              className="bg-gradient-to-r from-primary to-accent h-1.5 rounded-full transition-all duration-700 ease-out"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>

        {/* DP Budget */}
        <div className="glass-panel rounded-xl p-5 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Shield size={80} />
          </div>
          <p className="text-sm text-textMuted font-medium">Global DP Budget (ε)</p>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-4xl font-bold text-white">{sim.dpEpsilon.toFixed(2)}</span>
            <span className="text-sm text-textMuted">/ 5.00</span>
          </div>
          <div className="mt-4 flex items-center">
            <span className="w-1.5 h-1.5 rounded-full bg-secondary mr-2 animate-pulse" />
            <span className="text-xs text-secondary">
              {sim.dpEpsilon < 3 ? 'Privacy bounds secure' : 'Approaching budget limit'}
            </span>
          </div>
        </div>

        {/* Active Nodes */}
        <div className="glass-panel rounded-xl p-5 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Cpu size={80} />
          </div>
          <p className="text-sm text-textMuted font-medium">Active Hospitals</p>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-4xl font-bold text-white">{sim.activeNodes}</span>
            <span className="text-sm text-textMuted">Nodes</span>
          </div>
          <div className="flex -space-x-2 mt-4">
            {['NY', 'CH', 'SF', 'AU'].map((code, i) => (
              <div 
                key={code} 
                className={`w-7 h-7 rounded-full border-2 border-[#1E293B] flex items-center justify-center text-[10px] font-bold z-10 hover:z-20 transition-transform hover:scale-125 cursor-help ${
                  i < sim.activeNodes 
                    ? 'bg-gradient-to-br from-primary to-accent text-white' 
                    : 'bg-gray-700 text-gray-500'
                }`} 
                title={`Hospital ${code}`}
              >
                {code}
              </div>
            ))}
          </div>
        </div>

        {/* Current Accuracy */}
        <div className="glass-panel rounded-xl p-5 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Zap size={80} />
          </div>
          <p className="text-sm text-textMuted font-medium">Global Accuracy</p>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-4xl font-bold text-white">
              {sim.currentAccuracy > 0 ? (sim.currentAccuracy * 100).toFixed(1) : '—'}
            </span>
            {sim.currentAccuracy > 0 && <span className="text-sm text-textMuted">%</span>}
          </div>
          <div className="mt-4 flex items-center">
            {sim.currentAccuracy > 0 ? (
              <>
                <TrendingDown size={14} className="text-green-400 mr-1.5 rotate-180" />
                <span className="text-xs text-green-400">
                  +{(sim.currentAccuracy * 100 - 50).toFixed(1)}% from baseline
                </span>
              </>
            ) : (
              <span className="text-xs text-textMuted">Awaiting first round</span>
            )}
          </div>
        </div>
      </div>

      {/* Main Chart Area */}
      <div className="glass-panel rounded-xl p-6">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h2 className="text-lg font-semibold text-white">Global Model Convergence</h2>
            <p className="text-sm text-textMuted">Real-time loss tracking across federated rounds</p>
          </div>
          {sim.currentLoss > 0 && (
            <div className="text-right">
              <p className="text-sm text-textMuted">Current Loss</p>
              <p className="text-2xl font-mono text-primary font-bold">{sim.currentLoss.toFixed(4)}</p>
            </div>
          )}
        </div>
        
        <div className="h-[350px] w-full">
          <LossChart data={sim.lossHistory} />
        </div>
      </div>

      {/* Live Round Log */}
      <div className="glass-panel rounded-xl p-6">
        <div className="flex items-center space-x-3 mb-5">
          <BarChart3 className="text-accent" size={20} />
          <h2 className="text-lg font-semibold text-white">Live Round Log</h2>
          {sim.isRunning && (
            <span className="px-2 py-0.5 bg-green-500/10 text-green-400 border border-green-500/20 text-xs font-medium rounded-full animate-pulse">
              LIVE
            </span>
          )}
        </div>
        
        {sim.roundLog.length === 0 ? (
          <div className="text-center py-12 text-textMuted">
            <p>Click <strong>Start EHR Task</strong> or <strong>Start Imaging Task</strong> to begin a federated learning simulation.</p>
            <p className="text-sm mt-2">Rounds will appear here in real-time with loss, accuracy, and node participation data.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="pb-3 text-xs font-semibold text-textMuted uppercase tracking-wider">Round</th>
                  <th className="pb-3 text-xs font-semibold text-textMuted uppercase tracking-wider">Timestamp</th>
                  <th className="pb-3 text-xs font-semibold text-textMuted uppercase tracking-wider">Loss</th>
                  <th className="pb-3 text-xs font-semibold text-textMuted uppercase tracking-wider">Accuracy</th>
                  <th className="pb-3 text-xs font-semibold text-textMuted uppercase tracking-wider">Nodes</th>
                  <th className="pb-3 text-xs font-semibold text-textMuted uppercase tracking-wider">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {sim.roundLog.map((entry, idx) => (
                  <tr 
                    key={entry.round} 
                    className={`transition-all duration-300 ${idx === 0 && sim.isRunning ? 'bg-primary/5' : 'hover:bg-white/[0.02]'}`}
                  >
                    <td className="py-3 pr-6">
                      <span className="font-mono font-bold text-white">R{entry.round}</span>
                    </td>
                    <td className="py-3 pr-6 text-sm text-textMuted font-mono">{entry.timestamp}</td>
                    <td className="py-3 pr-6">
                      <span className="font-mono text-primary">{entry.loss.toFixed(4)}</span>
                    </td>
                    <td className="py-3 pr-6">
                      <span className="font-mono text-green-400">{(entry.accuracy * 100).toFixed(1)}%</span>
                    </td>
                    <td className="py-3 pr-6">
                      <span className={`text-sm font-medium ${entry.nodesReporting === 4 ? 'text-green-400' : 'text-yellow-400'}`}>
                        {entry.nodesReporting}/4
                      </span>
                    </td>
                    <td className="py-3">
                      <span className="px-2 py-1 text-xs rounded-full bg-green-500/10 text-green-400 border border-green-500/20">
                        Aggregated
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
