import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  Activity, 
  Network, 
  ShieldAlert, 
  Database, 
  Settings, 
  FileText, 
  Eye, 
  FlaskConical, 
  ShieldCheck, 
  Lock 
} from 'lucide-react';

const NAV_ITEMS = [
  { name: 'Dashboard', path: '/dashboard', icon: Activity },
  { name: 'Hospital Nodes', path: '/nodes', icon: Network },
  { name: 'Disease Prediction', path: '/prediction', icon: FileText },
  { name: 'Medical Imaging', path: '/imaging', icon: Eye },
  { name: 'Drug Discovery', path: '/drug', icon: FlaskConical },
  { name: 'Cyber Defense (IDS)', path: '/ids', icon: ShieldCheck },
  { name: 'Privacy & SecAgg', path: '/privacy', icon: Lock },
  { name: 'Global Settings', path: '/settings', icon: Settings },
];

export const Sidebar: React.FC = () => {
  return (
    <aside className="w-64 bg-surface border-r border-surfaceHighlight flex flex-col h-screen select-none">
      {/* Brand */}
      <div className="h-16 flex items-center px-6 border-b border-surfaceHighlight space-x-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-accent to-primary flex items-center justify-center text-white shadow-glow">
          <ShieldAlert className="w-5 h-5" />
        </div>
        <div>
          <span className="font-bold text-lg tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
            FedMedShield
          </span>
          <p className="text-[10px] text-textMuted tracking-wider font-mono uppercase">Multi-Task FL v2.0</p>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 p-4 space-y-1.5 overflow-y-auto">
        <p className="px-3 text-[11px] font-semibold text-textMuted tracking-wider uppercase mb-2">
          Orchestration & Modules
        </p>

        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center space-x-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-accent/15 text-accent font-semibold border-l-3 border-accent shadow-sm'
                    : 'text-textMuted hover:text-textMain hover:bg-surfaceHighlight/50'
                }`
              }
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              <span>{item.name}</span>
            </NavLink>
          );
        })}
      </nav>

      {/* Footer System Status */}
      <div className="p-4 border-t border-surfaceHighlight">
        <div className="p-3 rounded-lg bg-background border border-surfaceHighlight flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-xs font-medium text-textMain">All 4 Nodes Active</span>
          </div>
          <span className="text-[10px] font-mono text-textMuted">SecAgg On</span>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
