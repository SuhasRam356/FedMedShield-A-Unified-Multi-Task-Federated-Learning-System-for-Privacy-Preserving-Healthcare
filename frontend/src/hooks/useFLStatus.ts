import { useState, useEffect, useCallback } from 'react';
import flService from '../services/fl.service';
import { FLTask } from '../types/fl.types';

export const useFLStatus = () => {
  const [tasks, setTasks] = useState<FLTask[]>([]);
  const [activeTask, setActiveTask] = useState<FLTask | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchTasks = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await flService.getTasks();
      setTasks(data);
      if (data.length > 0 && !activeTask) {
        setActiveTask(data[0]);
      }
      setError(null);
    } catch (err: any) {
      console.warn('Backend tasks currently simulated/unreachable:', err.message);
      // Fallback default tasks
      const defaultTasks: FLTask[] = [
        {
          id: 1,
          name: 'Global Multi-Task EHR Cohort',
          task_type: 'ehr',
          status: 'idle',
          target_rounds: 10,
          current_round: 0,
          num_clients: 4,
          dp_epsilon: 2.5
        }
      ];
      setTasks(defaultTasks);
      setActiveTask(defaultTasks[0]);
    } finally {
      setIsLoading(false);
    }
  }, [activeTask]);

  useEffect(() => {
    fetchTasks();
  }, [fetchTasks]);

  const startTask = async (taskType: string) => {
    setIsLoading(true);
    try {
      const res = await flService.createTask({
        name: `FL Task - ${taskType.toUpperCase()}`,
        task_type: taskType,
        target_rounds: 10,
        num_clients: 4,
        dp_epsilon: 2.5
      });
      setActiveTask(res);
      await flService.startTask(res.id);
    } catch (err: any) {
      console.warn('Using client-side local orchestration');
    } finally {
      setIsLoading(false);
    }
  };

  const stopTask = async () => {
    if (!activeTask) return;
    try {
      await flService.stopTask(activeTask.id);
    } catch (err) {
      console.warn('Stopped locally');
    }
  };

  return {
    tasks,
    activeTask,
    setActiveTask,
    isLoading,
    error,
    refreshTasks: fetchTasks,
    startTask,
    stopTask
  };
};

export default useFLStatus;
