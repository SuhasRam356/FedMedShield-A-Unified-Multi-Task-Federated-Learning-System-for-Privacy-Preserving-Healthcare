/**
 * Utility functions for formatting data in the FedMedShield Dashboard
 */

export const formatMetric = (value: number | undefined | null, decimals: number = 4): string => {
    if (value === undefined || value === null) return "0.0000";
    return value.toFixed(decimals);
  };
  
  export const formatDPBudget = (round: number, baseEpsilonPerRound: number = 0.15): string => {
    return (round * baseEpsilonPerRound).toFixed(2);
  };
  
  export const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    }).format(date);
  };
  
  export const formatTaskName = (taskType: string): string => {
    const map: Record<string, string> = {
      'ehr': 'EHR Multi-Task',
      'imaging_tumor': 'Tumor Detection',
      'imaging_glaucoma': 'Glaucoma Detection',
      'drug': 'Drug Discovery',
      'ids': 'Intrusion Detection'
    };
    return map[taskType] || taskType.toUpperCase();
  };
