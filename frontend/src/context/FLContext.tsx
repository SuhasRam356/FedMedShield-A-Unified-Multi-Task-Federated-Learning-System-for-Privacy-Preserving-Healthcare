import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import { FLTask, FLMetrics, HospitalNode } from '../types/fl.types';
import flService from '../services/fl.service';

interface FLContextType {
  activeTask: FLTask | null;
  tasks: FLTask[];
  nodes: HospitalNode[];
  metrics: FLMetrics | null;
  roundHistory: { round: number; loss: number; accuracy: number }[];
  isOrchestrating: boolean;
  startOrchestration: (taskType: string) => Promise<void>;
  stopOrchestration: () => Promise<void>;
  setActiveTask: (task: FLTask | null) => void;
}

export const FLContext = createContext<FLContextType | undefined>(undefined);

export const FLProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [activeTask, setActiveTask] = useState<FLTask | null>(null);
  const [tasks, setTasks] = useState<FLTask[]>([]);
  const [nodes, setNodes] = useState<HospitalNode[]>([]);
  const [metrics, setMetrics] = useState<FLMetrics | null>(null);
  const [roundHistory, setRoundHistory] = useState<{ round: number; loss: number; accuracy: number }[]>([]);
  const [isOrchestrating, setIsOrchestrating] = useState<boolean>(false);

  // Initialize nodes and tasks
  useEffect(() => {
    const init = async () => {
      try {
        const fetchedNodes = await flService.getHospitalNodes();
        setNodes(fetchedNodes);
      } catch {
        setNodes([
          { client_id: 'hospital_ny', name: 'General Hospital, NY', region: 'US-East', status: 'online', data_size: '45.2 GB', latency_ms: 12, last_seen: 'Just now' },
          { client_id: 'hospital_chicago', name: 'Univ. Medical Center, Chicago', region: 'US-Central', status: 'online', data_size: '102.8 GB', latency_ms: 24, last_seen: 'Just now' },
          { client_id: 'hospital_sf', name: 'VA Medical Center, SF', region: 'US-West', status: 'online', data_size: '88.1 GB', latency_ms: 45, last_seen: 'Just now' },
          { client_id: 'hospital_austin', name: 'Community Clinic, Austin', region: 'US-South', status: 'online', data_size: '12.4 GB', latency_ms: 31, last_seen: 'Just now' },
        ]);
      }
    };
    init();
  }, []);

  const startOrchestration = useCallback(async (taskType: string) => {
    setIsOrchestrating(true);
    setRoundHistory([]);
    try {
      const task = await flService.createTask({
        name: `FL Task - ${taskType.toUpperCase()}`,
        task_type: taskType,
        target_rounds: 10,
        num_clients: 4,
        dp_epsilon: 2.5,
      });
      setActiveTask(task);
    } catch {
      setActiveTask({
        id: Date.now(),
        name: `FL Task - ${taskType.toUpperCase()}`,
        task_type: taskType,
        status: 'training',
        target_rounds: 10,
        current_round: 1,
        num_clients: 4,
        dp_epsilon: 2.5,
      });
    }
  }, []);

  const stopOrchestration = useCallback(async () => {
    if (activeTask) {
      try {
        await flService.stopTask(activeTask.id);
      } catch {
        // Stop local
      }
    }
    setIsOrchestrating(false);
  }, [activeTask]);

  return (
    <FLContext.Provider
      value={{
        activeTask,
        tasks,
        nodes,
        metrics,
        roundHistory,
        isOrchestrating,
        startOrchestration,
        stopOrchestration,
        setActiveTask,
      }}
    >
      {children}
    </FLContext.Provider>
  );
};

export const useFL = () => {
  const context = useContext(FLContext);
  if (!context) {
    throw new Error('useFL must be used within an FLProvider');
  }
  return context;
};
