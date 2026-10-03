import apiService from './api.service';
import { FLTask, FLTaskCreateRequest, HospitalNode } from '../types/fl.types';

export const flService = {
  getTasks: async (): Promise<FLTask[]> => {
    return apiService.get<FLTask[]>('/fl/tasks');
  },

  getTaskById: async (taskId: number): Promise<FLTask> => {
    return apiService.get<FLTask>(`/fl/tasks/${taskId}`);
  },

  createTask: async (data: FLTaskCreateRequest): Promise<FLTask> => {
    return apiService.post<FLTask>('/fl/tasks', data);
  },

  startTask: async (taskId: number): Promise<{ message: string; task: FLTask }> => {
    return apiService.post<{ message: string; task: FLTask }>(`/fl/tasks/${taskId}/start`);
  },

  stopTask: async (taskId: number): Promise<{ message: string; task: FLTask }> => {
    return apiService.post<{ message: string; task: FLTask }>(`/fl/tasks/${taskId}/stop`);
  },

  getHospitalNodes: async (): Promise<HospitalNode[]> => {
    return apiService.get<HospitalNode[]>('/fl/nodes');
  }
};

export default flService;
