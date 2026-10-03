/**
 * Global Constants and Configuration Thresholds
 * FedMedShield Multi-Task Federated Healthcare
 */

export const FL_CONFIG = {
  DEFAULT_ROUNDS: 10,
  DEFAULT_CLIENTS: 4,
  MAX_ROUNDS: 50,
  DEFAULT_DP_EPSILON: 2.5,
  MAX_DP_EPSILON: 10.0,
  DEFAULT_DP_DELTA: 1e-5,
  ROUND_DURATION_MS: 2000,
};

export const TASK_TYPES = [
  { id: 'ehr', label: 'EHR Multi-Task (Sepsis & COVID)', color: 'from-blue-500 to-indigo-500' },
  { id: 'imaging_tumor', label: 'ResNet18 Brain Tumor MRI', color: 'from-emerald-500 to-teal-500' },
  { id: 'imaging_glaucoma', label: 'Fundus Glaucoma Screening', color: 'from-cyan-500 to-blue-500' },
  { id: 'drug', label: 'Drug-Target Bioactivity Screening', color: 'from-purple-500 to-pink-500' },
  { id: 'ids', label: 'Network Cybersecurity IDS', color: 'from-amber-500 to-red-500' },
];

export const HOSPITALS = [
  { id: 'hospital_ny', name: 'General Hospital, NY', region: 'US-East', ip: '10.0.1.10' },
  { id: 'hospital_chicago', name: 'Univ. Medical Center, Chicago', region: 'US-Central', ip: '10.0.2.15' },
  { id: 'hospital_sf', name: 'VA Medical Center, SF', region: 'US-West', ip: '10.0.3.22' },
  { id: 'hospital_austin', name: 'Community Clinic, Austin', region: 'US-South', ip: '10.0.4.5' },
];

export const NAVIGATION_LINKS = [
  { name: 'Dashboard', path: '/dashboard', icon: 'Activity' },
  { name: 'Hospital Nodes', path: '/nodes', icon: 'Network' },
  { name: 'Disease Prediction', path: '/prediction', icon: 'FileText' },
  { name: 'Medical Imaging', path: '/imaging', icon: 'Eye' },
  { name: 'Drug Discovery', path: '/drug', icon: 'FlaskConical' },
  { name: 'Cyber Defense (IDS)', path: '/ids', icon: 'ShieldCheck' },
  { name: 'Privacy & SecAgg', path: '/privacy', icon: 'Lock' },
  { name: 'Settings', path: '/settings', icon: 'Sliders' },
];
