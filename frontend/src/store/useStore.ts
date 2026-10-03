import { create } from 'zustand';

interface Task {
  id: number;
  name: string;
  task_type: string;
  status: string;
  target_rounds: number;
}

interface WSMetrics {
  current_round: number;
  total_rounds: number;
  progress_percent: number;
  current_loss: number;
}

interface AppState {
  activeTask: Task | null;
  tasks: Task[];
  metrics: WSMetrics | null;
  setTasks: (tasks: Task[]) => void;
  setActiveTask: (task: Task | null) => void;
  updateMetrics: (metrics: WSMetrics) => void;
}

export const useStore = create<AppState>((set) => ({
  activeTask: null,
  tasks: [],
  metrics: null,
  setTasks: (tasks) => set({ tasks }),
  setActiveTask: (task) => set({ activeTask: task, metrics: null }),
  updateMetrics: (metrics) => set({ metrics }),
}));
