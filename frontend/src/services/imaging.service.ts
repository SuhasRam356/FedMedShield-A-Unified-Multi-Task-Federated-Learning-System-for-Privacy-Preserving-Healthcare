import apiService from './api.service';
import { ImagingPredictionResponse } from '../types/imaging.types';

export const imagingService = {
  analyzeScan: async (formData: FormData): Promise<any> => {
    return apiService.post('/imaging/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  analyzeScanJson: async (data: { modality: string; target_condition: string; patient_id?: string }): Promise<any> => {
    const form = new FormData();
    form.append('modality', data.modality);
    form.append('target_condition', data.target_condition);
    if (data.patient_id) form.append('patient_id', data.patient_id);
    return imagingService.analyzeScan(form);
  }
};

export default imagingService;
