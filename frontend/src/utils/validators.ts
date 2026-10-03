/**
 * Input validators for clinical EHR data, image formats, and network parameters
 * FedMedShield Framework
 */

export const validateVitalSigns = (vitals: {
  heartRate?: number;
  bloodPressureSystolic?: number;
  temperature?: number;
  oxygenSaturation?: number;
}): { isValid: boolean; errors: Record<string, string> } => {
  const errors: Record<string, string> = {};

  if (vitals.heartRate !== undefined && (vitals.heartRate < 30 || vitals.heartRate > 240)) {
    errors.heartRate = 'Heart rate must be between 30 and 240 bpm.';
  }

  if (vitals.bloodPressureSystolic !== undefined && (vitals.bloodPressureSystolic < 50 || vitals.bloodPressureSystolic > 260)) {
    errors.bloodPressureSystolic = 'Systolic BP must be between 50 and 260 mmHg.';
  }

  if (vitals.temperature !== undefined && (vitals.temperature < 32.0 || vitals.temperature > 43.0)) {
    errors.temperature = 'Temperature must be between 32.0°C and 43.0°C.';
  }

  if (vitals.oxygenSaturation !== undefined && (vitals.oxygenSaturation < 50 || vitals.oxygenSaturation > 100)) {
    errors.oxygenSaturation = 'Oxygen saturation must be between 50% and 100%.';
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors,
  };
};

export const validateSMILES = (smiles: string): boolean => {
  if (!smiles || smiles.trim().length === 0) return false;
  // Basic chemical SMILES syntax validation (balanced parentheses and valid atom characters)
  const validChars = /^[A-Za-z0-9@+\-\[\]\(\)\\=#$:./%]+$/;
  if (!validChars.test(smiles)) return false;

  let parenCount = 0;
  let bracketCount = 0;
  for (const char of smiles) {
    if (char === '(') parenCount++;
    if (char === ')') parenCount--;
    if (char === '[') bracketCount++;
    if (char === ']') bracketCount--;
    if (parenCount < 0 || bracketCount < 0) return false;
  }
  return parenCount === 0 && bracketCount === 0;
};

export const validateIPv4 = (ip: string): boolean => {
  const regex = /^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/;
  return regex.test(ip);
};
