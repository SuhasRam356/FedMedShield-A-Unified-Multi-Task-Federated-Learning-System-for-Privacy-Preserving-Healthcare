import { NavLink } from 'react-router-dom';
import { Activity, Shield, Network, Settings, Database, ActivitySquare } from 'lucide-react';
import clsx from 'clsx';

const Sidebar = () => {
  const navItems = [
    { name: 'Overview', path: '/dashboard', icon: Activity },
    { name: 'FL Network', path: '/network', icon: Network },
    { name: 'Privacy & SecAgg', path: '/privacy', icon: Shield },
    { name: 'Data Nodes', path: '/nodes', icon: Database },
    { name: 'Settings', path: '/settings', icon: Settings },
  ];

  return (
    <aside className="w-64 bg-card/40 border-r border-white/10 flex flex-col h-full backdrop-blur-xl transition-all">
      <div className="p-6 flex items-center space-x-3 border-b border-white/5">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center shadow-lg shadow-primary/20">
          <ActivitySquare size={20} className="text-white" />
        </div>
        <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-textMuted tracking-tight">
          FedMedShield
        </span>
      </div>

      <nav className="flex-1 py-6 px-3 space-y-1 overflow-y-auto">
        <div className="text-xs font-semibold text-textMuted uppercase tracking-wider mb-4 px-3">
          Orchestration
        </div>
        
        {navItems.map((item) => (
          <NavLink
            key={item.name}
            to={item.path}
            className={({ isActive }) => clsx(
              "flex items-center space-x-3 px-3 py-2.5 rounded-lg transition-all duration-200 group",
              isActive 
                ? "bg-primary/10 text-primary font-medium" 
                : "text-textMuted hover:bg-white/5 hover:text-textMain"
            )}
          >
            <item.icon size={18} className={clsx("transition-colors", "group-hover:text-primary")} />
            <span>{item.name}</span>
          </NavLink>
        ))}
      </nav>

      <div className="p-4 border-t border-white/5">
        <div className="bg-gradient-to-r from-primary/20 to-accent/20 p-4 rounded-xl border border-primary/20">
          <p className="text-sm font-medium text-white">System Status</p>
          <div className="flex items-center space-x-2 mt-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-secondary opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-secondary"></span>
            </span>
            <span className="text-xs text-textMuted">All nodes operational</span>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
