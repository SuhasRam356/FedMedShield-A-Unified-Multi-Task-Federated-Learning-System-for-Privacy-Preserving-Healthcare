import { Bell, Search, User } from 'lucide-react';

const Header = () => {
  return (
    <header className="h-16 border-b border-white/5 bg-card/20 backdrop-blur-md flex items-center justify-between px-6 sticky top-0 z-10">
      <div className="flex items-center bg-background/50 border border-white/10 rounded-lg px-3 py-1.5 w-64 focus-within:border-primary/50 transition-colors">
        <Search size={16} className="text-textMuted mr-2" />
        <input 
          type="text" 
          placeholder="Search logs, nodes..." 
          className="bg-transparent border-none outline-none text-sm w-full text-textMain placeholder:text-textMuted/50"
        />
      </div>

      <div className="flex items-center space-x-4">
        <button className="relative p-2 rounded-full hover:bg-white/5 transition-colors text-textMuted hover:text-textMain">
          <Bell size={18} />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-accent"></span>
        </button>
        
        <div className="h-6 w-px bg-white/10 mx-2"></div>
        
        <div className="flex items-center space-x-3 cursor-pointer group">
          <div className="text-right hidden sm:block">
            <p className="text-sm font-medium text-textMain group-hover:text-primary transition-colors">Dr. Smith</p>
            <p className="text-xs text-textMuted">Lead Researcher</p>
          </div>
          <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-primary to-accent flex items-center justify-center p-0.5">
            <div className="w-full h-full rounded-full bg-card flex items-center justify-center">
              <User size={16} className="text-white" />
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
