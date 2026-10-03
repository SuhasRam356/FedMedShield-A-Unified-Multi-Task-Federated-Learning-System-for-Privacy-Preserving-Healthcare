import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';

interface AccuracyChartProps {
  data: { round: number; accuracy: number }[];
}

export const AccuracyChart: React.FC<AccuracyChartProps> = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center border border-dashed border-white/10 rounded-xl text-textMuted text-sm">
        Awaiting round evaluation...
      </div>
    );
  }

  const formattedData = data.map(d => ({
    round: `R${d.round}`,
    accuracyPct: +(d.accuracy * 100).toFixed(2)
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={formattedData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
        <XAxis dataKey="round" stroke="#94A3B8" tickLine={false} tick={{ fontSize: 12 }} />
        <YAxis stroke="#94A3B8" tickLine={false} tick={{ fontSize: 12 }} domain={[40, 100]} />
        <Tooltip
          contentStyle={{ backgroundColor: '#1E293B', borderColor: '#334155', borderRadius: '8px' }}
          formatter={(value: any) => [`${value}%`, 'Global Accuracy']}
        />
        <Line
          type="monotone"
          dataKey="accuracyPct"
          stroke="#10B981"
          strokeWidth={3}
          dot={{ r: 4, fill: '#10B981' }}
          activeDot={{ r: 6 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default AccuracyChart;
