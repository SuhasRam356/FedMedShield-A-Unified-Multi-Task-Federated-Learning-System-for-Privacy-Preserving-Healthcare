import React from 'react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  label?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ size = 'md', label }) => {
  const sizeMap = {
    sm: 'w-4 h-4 border-2',
    md: 'w-8 h-8 border-3',
    lg: 'w-12 h-12 border-4',
  };

  return (
    <div className="flex flex-col items-center justify-center space-y-3">
      <div
        className={`${sizeMap[size]} border-primary border-t-transparent rounded-full animate-spin`}
      />
      {label && <p className="text-sm text-textMuted font-medium">{label}</p>}
    </div>
  );
};

export default LoadingSpinner;
