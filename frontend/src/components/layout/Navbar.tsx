import React from 'react';
import { Bell, Search, Shield, User } from 'lucide-react';
import { useAuth } from '../../hooks/useAuth';

export const Navbar: React.FC = () => {
  const { user } = useAuth();

  return (
    <header className="h-16 border-b border-surfaceHighlight bg-surface/50 backdrop-blur-md px-6 flex items-center justify-between z-10">
      {/* Search Bar */}
      <div className="relative w-80">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-textMuted w-4 h-4" />
        <input
          type="text"
          placeholder="Search clinical models, logs, nodes..."
          className="w-full bg-background border border-surfaceHighlight rounded-lg pl-9 pr-4 py-1.5 text-sm text-textMain placeholder-textMuted focus:outline-none focus:border-accent transition-colors"
        />
      </div>

      {/* Right Controls */}
      <div className="flex items-center space-x-5">
        <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-xs font-medium">
          <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <span>FL Network Secure</span>
        </div>

        <button className="relative p-2 text-textMuted hover:text-textMain transition-colors">
          <Bell className="w-5 h-5" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-accent" />
        </button>

        <div className="flex items-center space-x-3 pl-3 border-l border-surfaceHighlight">
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-accent to-primary flex items-center justify-center text-white font-bold text-xs shadow-md">
            {user?.username ? user.username.substring(0, 2).toUpperCase() : 'DS'}
          </div>
          <div className="text-left hidden md:block">
            <p className="text-xs font-semibold text-textMain">{user?.username || 'Dr. Smith'}</p>
            <p className="text-[10px] text-textMuted capitalize">{user?.role || 'Lead Researcher'}</p>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
