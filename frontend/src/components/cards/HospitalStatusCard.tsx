import React from 'react';
import { HospitalNode } from '../../types/fl.types';
import { HardDrive, Signal, SignalZero, ShieldCheck } from 'lucide-react';

interface HospitalStatusCardProps {
  node: HospitalNode;
}

export const HospitalStatusCard: React.FC<HospitalStatusCardProps> = ({ node }) => {
  const isOnline = node.status === 'online';

  return (
    <div className="bg-surface border border-surfaceHighlight rounded-xl p-5 shadow-glow hover:border-accent/40 transition-all flex flex-col justify-between">
      <div>
        <div className="flex items-start justify-between">
          <div>
            <h3 className="font-semibold text-textMain text-base">{node.name}</h3>
            <p className="text-xs text-textMuted font-mono mt-0.5">{node.region} • {node.client_id}</p>
          </div>
          <span
            className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${
              isOnline
                ? 'bg-green-500/10 text-green-400 border-green-500/20'
                : 'bg-red-500/10 text-red-400 border-red-500/20'
            }`}
          >
            {isOnline ? (
              <>
                <Signal className="w-3 h-3 mr-1" /> Online
              </>
            ) : (
              <>
                <SignalZero className="w-3 h-3 mr-1" /> Offline
              </>
            )}
          </span>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
          <div className="p-2.5 bg-background rounded-lg border border-surfaceHighlight">
            <span className="text-textMuted flex items-center mb-1">
              <HardDrive className="w-3.5 h-3.5 mr-1 text-primary" /> Dataset
            </span>
            <span className="font-mono font-medium text-textMain">{node.data_size}</span>
          </div>

          <div className="p-2.5 bg-background rounded-lg border border-surfaceHighlight">
            <span className="text-textMuted flex items-center mb-1">
              <Signal className="w-3.5 h-3.5 mr-1 text-secondary" /> Ping
            </span>
            <span className="font-mono font-medium text-textMain">{node.latency_ms} ms</span>
          </div>
        </div>
      </div>

      <div className="mt-4 pt-3 border-t border-surfaceHighlight flex items-center justify-between text-xs text-textMuted">
        <span className="flex items-center text-green-400">
          <ShieldCheck className="w-3.5 h-3.5 mr-1" /> SecAgg Enforced
        </span>
        <span className="font-mono">{node.last_seen}</span>
      </div>
    </div>
  );
};

export default HospitalStatusCard;
