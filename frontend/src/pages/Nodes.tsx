import React from 'react';
import { Database, Signal, SignalZero, HardDrive } from 'lucide-react';

const DUMMY_NODES = [
  { id: 'h1', name: 'General Hospital, NY', status: 'online', dataSize: '45.2 GB', ping: '12ms' },
  { id: 'h2', name: 'Univ. Medical Center, Chicago', status: 'online', dataSize: '102.8 GB', ping: '24ms' },
  { id: 'h3', name: 'VA Hospital, SF', status: 'online', dataSize: '88.1 GB', ping: '45ms' },
  { id: 'h4', name: 'Community Clinic, Austin', status: 'offline', dataSize: '12.4 GB', ping: '-' },
];

const Nodes: React.FC = () => {
  return (
    <div className="space-y-6 fade-in">
      <div>
        <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-500">
          Data Nodes
        </h1>
        <p className="text-textMuted mt-1">
          Manage and monitor decentralized hospital nodes participating in the federation.
        </p>
      </div>

      <div className="bg-surface border border-surfaceHighlight rounded-xl overflow-hidden shadow-glow">
        <div className="p-6 border-b border-surfaceHighlight flex justify-between items-center">
          <h2 className="text-xl font-semibold flex items-center">
            <Database className="mr-3 text-blue-400" />
            Registered Institutions
          </h2>
          <button className="px-4 py-2 bg-background border border-surfaceHighlight rounded-lg text-sm font-medium hover:border-accent transition-colors">
            + Provision New Node
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead className="bg-background">
              <tr>
                <th className="px-6 py-4 text-sm font-semibold text-textMuted">Institution Name</th>
                <th className="px-6 py-4 text-sm font-semibold text-textMuted">Status</th>
                <th className="px-6 py-4 text-sm font-semibold text-textMuted">Local Dataset</th>
                <th className="px-6 py-4 text-sm font-semibold text-textMuted">Latency</th>
                <th className="px-6 py-4 text-sm font-semibold text-textMuted text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surfaceHighlight">
              {DUMMY_NODES.map((node) => (
                <tr key={node.id} className="hover:bg-background/50 transition-colors">
                  <td className="px-6 py-4">
                    <div className="font-medium text-textMain">{node.name}</div>
                    <div className="text-xs text-textMuted font-mono mt-1">ID: {node.id.toUpperCase()}</div>
                  </td>
                  <td className="px-6 py-4">
                    {node.status === 'online' ? (
                      <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-green-500/10 text-green-400 border border-green-500/20">
                        <Signal className="w-3 h-3 mr-1" /> Online
                      </span>
                    ) : (
                      <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-red-500/10 text-red-400 border border-red-500/20">
                        <SignalZero className="w-3 h-3 mr-1" /> Offline
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center text-sm text-textMuted">
                      <HardDrive className="w-4 h-4 mr-2 text-indigo-400" />
                      {node.dataSize}
                    </div>
                  </td>
                  <td className="px-6 py-4 text-sm text-textMuted">
                    {node.ping}
                  </td>
                  <td className="px-6 py-4 text-right">
                    <button className="text-accent hover:text-accentHover text-sm font-medium transition-colors">
                      Configure
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Nodes;
