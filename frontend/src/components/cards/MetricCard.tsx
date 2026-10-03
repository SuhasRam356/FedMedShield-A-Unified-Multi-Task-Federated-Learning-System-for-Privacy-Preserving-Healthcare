import React, { ReactNode } from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  change?: string;
  isPositive?: boolean;
  icon?: ReactNode;
  gradient?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  change,
  isPositive = true,
  icon,
  gradient = 'from-primary/10 to-transparent'
}) => {
  return (
    <div className={`relative overflow-hidden rounded-xl border border-surfaceHighlight bg-surface p-5 shadow-glow transition-all hover:border-accent/40 bg-gradient-to-br ${gradient}`}>
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-textMuted">{title}</p>
        {icon && <div className="text-accent">{icon}</div>}
      </div>

      <div className="mt-3 flex items-baseline space-x-2">
        <span className="text-3xl font-bold font-mono tracking-tight text-white">{value}</span>
      </div>

      <div className="mt-3 flex items-center justify-between text-xs">
        {subtitle && <span className="text-textMuted">{subtitle}</span>}
        {change && (
          <span className={`font-semibold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
            {change}
          </span>
        )}
      </div>
    </div>
  );
};

export default MetricCard;
