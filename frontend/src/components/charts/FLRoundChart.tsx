import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';

interface FLRoundChartProps {
  data: { round: number; durationSeconds: number; clientCount: number }[];
}

export const FLRoundChart: React.FC<FLRoundChartProps> = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center border border-dashed border-white/10 rounded-xl text-textMuted text-sm">
        No federated rounds recorded yet.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
        <XAxis dataKey="round" stroke="#94A3B8" tickLine={false} tick={{ fontSize: 12 }} />
        <YAxis stroke="#94A3B8" tickLine={false} tick={{ fontSize: 12 }} />
        <Tooltip
          contentStyle={{ backgroundColor: '#1E293B', borderColor: '#334155', borderRadius: '8px' }}
          itemStyle={{ color: '#60A5FA' }}
        />
        <Bar dataKey="durationSeconds" name="Round Duration (s)" radius={[4, 4, 0, 0]}>
          {data.map((_, index) => (
            <Cell key={`cell-${index}`} fill={index % 2 === 0 ? '#3B82F6' : '#6366F1'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

export default FLRoundChart;
