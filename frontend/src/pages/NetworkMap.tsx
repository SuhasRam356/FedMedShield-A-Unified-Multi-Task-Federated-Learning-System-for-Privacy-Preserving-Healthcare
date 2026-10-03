import { Server, Database, Activity, Lock, Share2 } from 'lucide-react';
import { useStore } from '../store/useStore';
import clsx from 'clsx';

const NetworkMap = () => {
  const { metrics, activeTask } = useStore();
  const isRunning = !!activeTask;

  const hospitals = [
    { id: 'hospital-a', name: 'General Hospital', location: 'New York', color: 'from-blue-500 to-cyan-500' },
    { id: 'hospital-b', name: 'Cancer Institute', location: 'Boston', color: 'from-purple-500 to-pink-500' },
    { id: 'hospital-c', name: 'Univ. Medical Center', location: 'Chicago', color: 'from-orange-500 to-amber-500' },
    { id: 'hospital-d', name: 'VA Hospital', location: 'San Francisco', color: 'from-emerald-500 to-teal-500' },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white">FL Network Topology</h1>
        <p className="text-textMuted mt-1">Live visualization of the Secure Aggregation network and data nodes.</p>
      </div>

      <div className="glass-panel rounded-xl p-8 relative min-h-[600px] flex flex-col items-center justify-center overflow-hidden">
        
        {/* Background Grid Pattern */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff05_1px,transparent_1px),linear-gradient(to_bottom,#ffffff05_1px,transparent_1px)] bg-[size:4rem_4rem]"></div>

        {/* Central Server */}
        <div className="relative z-10 flex flex-col items-center">
          <div className={clsx(
            "w-32 h-32 rounded-full flex flex-col items-center justify-center border-4 shadow-2xl transition-all duration-1000 z-20 bg-card",
            isRunning ? "border-primary shadow-primary/20" : "border-white/10 shadow-black/50"
          )}>
            <Server size={32} className={clsx("mb-2 transition-colors", isRunning ? "text-primary" : "text-textMuted")} />
            <span className="font-bold text-white text-sm">Central Server</span>
            <span className="text-[10px] text-textMuted">Global Model</span>
          </div>
          
          {/* Animated rings around server */}
          {isRunning && (
            <>
              <div className="absolute w-40 h-40 border border-primary/30 rounded-full animate-ping opacity-20"></div>
              <div className="absolute w-48 h-48 border border-accent/20 rounded-full animate-[spin_10s_linear_infinite] border-t-accent"></div>
            </>
          )}
        </div>

        {/* Hospitals Ring */}
        <div className="absolute w-[500px] h-[500px] rounded-full border border-dashed border-white/10 animate-[spin_60s_linear_infinite]"></div>

        {/* Hospital Nodes */}
        {hospitals.map((hospital, index) => {
          // Calculate positions in a circle
          const angle = (index * (360 / hospitals.length)) * (Math.PI / 180);
          const radius = 250; // Distance from center
          const x = Math.cos(angle) * radius;
          const y = Math.sin(angle) * radius;

          return (
            <div 
              key={hospital.id}
              className="absolute z-20 flex flex-col items-center group cursor-pointer"
              style={{
                transform: `translate(${x}px, ${y}px)`,
              }}
            >
              <div className="relative">
                {/* Connection Line to center (SVG) */}
                <svg className="absolute pointer-events-none top-1/2 left-1/2 -z-10 overflow-visible" style={{ width: 0, height: 0 }}>
                  <line 
                    x1={0} 
                    y1={0} 
                    x2={-x} 
                    y2={-y} 
                    stroke={isRunning ? '#3B82F6' : '#ffffff20'} 
                    strokeWidth={isRunning ? 2 : 1}
                    strokeDasharray={isRunning ? "5,5" : "0"}
                    className={isRunning ? "animate-[dash_2s_linear_infinite]" : ""}
                  />
                </svg>
                
                {/* Node Icon */}
                <div className={clsx(
                  "w-16 h-16 rounded-2xl bg-gradient-to-br p-0.5 shadow-lg transition-transform group-hover:scale-110",
                  hospital.color
                )}>
                  <div className="w-full h-full bg-card rounded-[14px] flex items-center justify-center">
                    <Database size={24} className="text-white" />
                  </div>
                </div>

                {/* Status indicator */}
                <div className={clsx(
                  "absolute -top-1 -right-1 w-4 h-4 rounded-full border-2 border-card",
                  isRunning ? "bg-secondary" : "bg-textMuted"
                )}></div>
              </div>

              {/* Node Details (always facing up, compensating for any rotation if we added it) */}
              <div className="mt-3 text-center bg-card/80 backdrop-blur px-3 py-1.5 rounded-lg border border-white/5 shadow-xl">
                <p className="text-sm font-bold text-white whitespace-nowrap">{hospital.name}</p>
                <div className="flex items-center justify-center space-x-2 mt-1">
                  <span className="text-xs text-textMuted flex items-center"><Activity size={10} className="mr-1"/> {hospital.location}</span>
                </div>
                {isRunning && (
                  <div className="flex items-center justify-center space-x-2 mt-2 pt-2 border-t border-white/10">
                     <span className="text-[10px] text-accent flex items-center bg-accent/10 px-1.5 rounded"><Lock size={10} className="mr-1"/> SecAgg</span>
                     <span className="text-[10px] text-primary flex items-center bg-primary/10 px-1.5 rounded"><Share2 size={10} className="mr-1"/> DP-SGD</span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <style>{`
        @keyframes dash {
          to {
            stroke-dashoffset: -20;
          }
        }
      `}</style>
    </div>
  );
};

export default NetworkMap;
