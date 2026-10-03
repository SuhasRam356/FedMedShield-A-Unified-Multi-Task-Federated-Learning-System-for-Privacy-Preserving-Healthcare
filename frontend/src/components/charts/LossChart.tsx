import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';

interface LossChartProps {
  data: { round: number; loss: number }[];
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-card/90 backdrop-blur border border-white/10 p-3 rounded-lg shadow-xl">
        <p className="text-sm text-textMuted mb-1">Round {label}</p>
        <p className="text-primary font-mono font-bold">
          Loss: {payload[0].value.toFixed(4)}
        </p>
      </div>
    );
  }
  return null;
};

const LossChart = ({ data }: LossChartProps) => {
  if (!data || data.length === 0) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center border border-dashed border-white/10 rounded-lg">
        <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mb-4">
          <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
        </div>
        <p className="text-textMuted">Waiting for orchestration to begin...</p>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart
        data={data}
        margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
      >
        <defs>
          <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
        <XAxis 
          dataKey="round" 
          stroke="#94A3B8" 
          tick={{ fill: '#94A3B8', fontSize: 12 }}
          tickLine={false}
          axisLine={false}
          dy={10}
        />
        <YAxis 
          stroke="#94A3B8" 
          tick={{ fill: '#94A3B8', fontSize: 12, fontFamily: 'monospace' }}
          tickLine={false}
          axisLine={false}
          dx={-10}
          domain={['auto', 'auto']}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#ffffff20', strokeWidth: 1, strokeDasharray: '5 5' }} />
        <Area
          type="monotone"
          dataKey="loss"
          stroke="#3B82F6"
          strokeWidth={3}
          fillOpacity={1}
          fill="url(#colorLoss)"
          isAnimationActive={true}
          animationDuration={500}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
};

export default LossChart;
