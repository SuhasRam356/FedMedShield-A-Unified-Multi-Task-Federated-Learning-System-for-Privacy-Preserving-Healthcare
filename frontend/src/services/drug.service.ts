import apiService from './api.service';

export interface CompoundScreenRequest {
  compound_id: string;
  smiles_string: string;
  target_protein: string;
  molecular_weight?: number;
}

export interface CompoundScreenResponse {
  compound_id: string;
  target_protein: string;
  predicted_kd_nm: number;
  binding_affinity_score: number;
  bioactivity_class: string;
  druggability_probability: number;
  safety_admet_flag: string;
  recommendation: string;
}

export const drugService = {
  screenCompound: async (data: CompoundScreenRequest): Promise<CompoundScreenResponse> => {
    return apiService.post<CompoundScreenResponse>('/drug/screen', data);
  }
};

export default drugService;
