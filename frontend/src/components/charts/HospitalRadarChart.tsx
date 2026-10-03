import React from 'react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip
} from 'recharts';

interface HospitalRadarProps {
  data?: { metric: string; NY: number; Chicago: number; SF: number; Austin: number }[];
}

const DEFAULT_RADAR_DATA = [
  { metric: 'Data Size', NY: 80, Chicago: 95, SF: 70, Austin: 50 },
  { metric: 'Accuracy', NY: 90, Chicago: 92, SF: 88, Austin: 85 },
  { metric: 'Low Latency', NY: 95, Chicago: 85, SF: 65, Austin: 75 },
  { metric: 'Completeness', NY: 85, Chicago: 90, SF: 92, Austin: 80 },
  { metric: 'Compliance', NY: 100, Chicago: 100, SF: 100, Austin: 100 },
];

export const HospitalRadarChart: React.FC<HospitalRadarProps> = ({ data = DEFAULT_RADAR_DATA }) => {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <RadarChart data={data}>
        <PolarGrid stroke="#ffffff15" />
        <PolarAngleAxis dataKey="metric" tick={{ fill: '#94A3B8', fontSize: 11 }} />
        <PolarRadiusAxis stroke="#ffffff10" tick={{ fill: '#94A3B8', fontSize: 10 }} />
        <Tooltip
          contentStyle={{ backgroundColor: '#1E293B', borderColor: '#334155', borderRadius: '8px' }}
        />
        <Radar name="General Hospital (NY)" dataKey="NY" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.2} />
        <Radar name="Univ. Med (Chicago)" dataKey="Chicago" stroke="#10B981" fill="#10B981" fillOpacity={0.2} />
      </RadarChart>
    </ResponsiveContainer>
  );
};

export default HospitalRadarChart;
