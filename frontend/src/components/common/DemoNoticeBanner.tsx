import React from 'react';
import { AlertTriangle } from 'lucide-react';

interface DemoNoticeBannerProps {
  title?: string;
  message: string;
  variant?: 'warning' | 'info' | 'critical';
}

export const DemoNoticeBanner: React.FC<DemoNoticeBannerProps> = ({
  title = 'Research Demonstration Prototype — Not for Clinical or Diagnostic Use',
  message,
  variant = 'warning',
}) => {
  const styles = {
    warning: 'bg-amber-500/10 border-amber-500/30 text-amber-300',
    critical: 'bg-red-500/10 border-red-500/30 text-red-300',
    info: 'bg-blue-500/10 border-blue-500/30 text-blue-300',
  }[variant];

  return (
    <div
      role="alert"
      className={`w-full rounded-xl border p-4 shadow-sm backdrop-blur-sm flex items-start space-x-3.5 ${styles}`}
    >
      <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />
      <div className="text-xs leading-relaxed space-y-0.5">
        <strong className="font-semibold uppercase tracking-wider block text-white/95">
          {title}
        </strong>
        <p className="opacity-90">{message}</p>
      </div>
    </div>
  );
};

export default DemoNoticeBanner;
