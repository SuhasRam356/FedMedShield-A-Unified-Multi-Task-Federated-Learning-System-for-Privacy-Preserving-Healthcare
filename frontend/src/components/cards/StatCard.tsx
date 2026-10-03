import React from 'react';
import { LucideIcon } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  subValue?: string | number;
  icon: LucideIcon;
  progress?: number;
  statusText?: React.ReactNode;
}

const StatCard: React.FC<StatCardProps> = ({ 
  title, 
  value, 
  subtitle, 
  subValue, 
  icon: Icon, 
  progress, 
  statusText 
}) => {
  return (
    <div className="glass-panel rounded-xl p-5 relative overflow-hidden group">
      <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
        <Icon size={80} />
      </div>
      
      <p className="text-sm text-textMuted font-medium">{title}</p>
      
      <div className="mt-2 flex items-baseline space-x-2">
        <span className="text-4xl font-bold text-white">{value}</span>
        {subValue && (
          <span className="text-sm text-textMuted">/ {subValue}</span>
        )}
      </div>
      
      {progress !== undefined && (
        <div className="w-full bg-background rounded-full h-1.5 mt-4 overflow-hidden border border-white/5">
          <div 
            className="bg-gradient-to-r from-primary to-accent h-1.5 rounded-full transition-all duration-500"
            style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
          ></div>
        </div>
      )}
      
      {subtitle && !progress && !statusText && (
        <p className="text-sm text-textMuted mt-4">{subtitle}</p>
      )}
      
      {statusText && (
        <div className="mt-4 text-xs font-medium">
          {statusText}
        </div>
      )}
    </div>
  );
};

export default StatCard;
