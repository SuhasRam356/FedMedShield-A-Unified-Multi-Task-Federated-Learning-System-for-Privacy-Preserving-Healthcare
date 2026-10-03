import apiService from './api.service';
import { PatientData, PredictionResponse, DiseasePrediction, SepsisAlert } from '../types/prediction.types';

export const predictionService = {
  predictEHRRisk: async (patientData: Partial<PatientData>): Promise<any> => {
    return apiService.post('/prediction/ehr', {
      age: patientData.age || 58,
      gender: patientData.gender || 'male',
      heart_rate: patientData.heartRate || 95.0,
      systolic_bp: patientData.bloodPressureSystolic || 120.0,
      diastolic_bp: patientData.bloodPressureDiastolic || 80.0,
      respiratory_rate: patientData.respiratoryRate || 20.0,
      temperature: patientData.temperature || 38.2,
      spo2: patientData.oxygenSaturation || 94.0,
      white_blood_cell: patientData.wbc || 13.5,
      platelets: 185.0,
      creatinine: patientData.creatinine || 1.3,
      crp: 42.0,
      comorbidities: ['Hypertension', 'Diabetes']
    });
  },

  getHistoricalPredictions: async (): Promise<DiseasePrediction[]> => {
    return apiService.get<DiseasePrediction[]>('/prediction/history');
  }
};

export default predictionService;
