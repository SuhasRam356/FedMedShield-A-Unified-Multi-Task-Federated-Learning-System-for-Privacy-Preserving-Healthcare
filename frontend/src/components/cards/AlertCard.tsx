import React from 'react';
import { AlertTriangle, ShieldAlert, CheckCircle, Info } from 'lucide-react';

interface AlertCardProps {
  title: string;
  message: string;
  severity?: 'info' | 'warning' | 'error' | 'success';
  timestamp?: string;
}

export const AlertCard: React.FC<AlertCardProps> = ({
  title,
  message,
  severity = 'info',
  timestamp
}) => {
  const styles = {
    info: 'border-blue-500/20 bg-blue-500/10 text-blue-400',
    warning: 'border-yellow-500/20 bg-yellow-500/10 text-yellow-400',
    error: 'border-red-500/20 bg-red-500/10 text-red-400',
    success: 'border-green-500/20 bg-green-500/10 text-green-400',
  };

  const icons = {
    info: <Info className="w-5 h-5" />,
    warning: <AlertTriangle className="w-5 h-5" />,
    error: <ShieldAlert className="w-5 h-5" />,
    success: <CheckCircle className="w-5 h-5" />,
  };

  return (
    <div className={`p-4 rounded-xl border flex items-start space-x-3.5 ${styles[severity]}`}>
      <div className="flex-shrink-0 mt-0.5">{icons[severity]}</div>
      <div className="flex-1">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-semibold text-textMain">{title}</h4>
          {timestamp && <span className="text-[10px] font-mono opacity-60">{timestamp}</span>}
        </div>
        <p className="text-xs text-textMuted mt-1 leading-relaxed">{message}</p>
      </div>
    </div>
  );
};

export default AlertCard;
