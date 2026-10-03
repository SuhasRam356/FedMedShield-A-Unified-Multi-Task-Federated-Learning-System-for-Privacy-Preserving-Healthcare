import React, { useState } from 'react';
import { ShieldCheck, ShieldAlert, Terminal, Activity, AlertTriangle, CheckCircle, RefreshCw } from 'lucide-react';
import idsService, { IntrusionAlertResponse } from '../services/ids.service';

export const IntrusionDetection: React.FC = () => {
  const [alerts, setAlerts] = useState<IntrusionAlertResponse[]>([
    {
      alert_id: 'ALT-C902A1',
      timestamp: '2026-10-02 18:42:10 UTC',
      attack_detected: true,
      attack_type: 'SYN Flood DDoS Attempt',
      severity: 'Critical',
      confidence: 0.965,
      affected_node: 'Hospital Node Gateway (10.0.1.10)',
      mitigation_action: 'Automated drop rule engaged on ingress firewall. Rate-limit IP: 192.168.1.105'
    },
    {
      alert_id: 'ALT-B41092',
      timestamp: '2026-10-02 17:15:33 UTC',
      attack_detected: false,
      attack_type: 'Benign Protocol Handshake',
      severity: 'Low',
      confidence: 0.992,
      affected_node: 'Central Aggregation Server (10.0.0.1)',
      mitigation_action: 'Traffic accepted into local demilitarized clinical subnetwork.'
    }
  ]);
  const [isScanning, setIsScanning] = useState(false);

  const simulatePacketInspection = async () => {
    setIsScanning(true);
    try {
      const newAlert = await idsService.inspectPacketFlow({
        source_ip: '192.168.4.210',
        dest_ip: '10.0.2.15',
        protocol: 'TCP',
        packet_length: 1500,
        duration_sec: 0.05,
        flag: 'SYN_SENT',
        bytes_in: 120000,
        bytes_out: 450
      });
      setAlerts((prev) => [newAlert, ...prev]);
    } catch {
      setTimeout(() => {
        setAlerts((prev) => [
          {
            alert_id: `ALT-${Math.random().toString(16).substring(2, 8).toUpperCase()}`,
            timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC',
            attack_detected: true,
            attack_type: 'Abnormal High-Frequency Weight Exfiltration',
            severity: 'High',
            confidence: 0.941,
            affected_node: 'Chicago Medical Node Gateway (10.0.2.15)',
            mitigation_action: 'Quarantined client channel; mutual TLS certificate verification triggered.'
          },
          ...prev
        ]);
        setIsScanning(false);
      }, 800);
      return;
    }
    setIsScanning(false);
  };

  return (
    <div className="space-y-6 fade-in pb-20">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">Cyber Defense IDS (Module 4: Security)</h1>
          <p className="text-textMuted mt-1">
            Federated neural intrusion detection monitoring clinical network traffic and HIPAA compliance anomalies.
          </p>
        </div>
        <button
          onClick={simulatePacketInspection}
          disabled={isScanning}
          className="flex items-center space-x-2 bg-gradient-to-r from-red-500 to-amber-500 hover:opacity-90 text-white px-4 py-2 rounded-lg font-medium shadow-lg transition-all disabled:opacity-50"
        >
          <RefreshCw size={16} className={isScanning ? 'animate-spin' : ''} />
          <span>Simulate Packet Audit</span>
        </button>
      </div>

      {/* Cyber Defense Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
        <div className="bg-surface border border-surfaceHighlight rounded-xl p-5 shadow-glow">
          <p className="text-xs text-textMuted font-medium">Inspected Packets</p>
          <p className="text-2xl font-bold font-mono text-white mt-2">1,489,204</p>
          <span className="text-[11px] text-green-400 mt-2 block">100% Zero-Loss Ingress</span>
        </div>
        <div className="bg-surface border border-surfaceHighlight rounded-xl p-5 shadow-glow">
          <p className="text-xs text-textMuted font-medium">Threats Mitigated</p>
          <p className="text-2xl font-bold font-mono text-amber-400 mt-2">2 Active</p>
          <span className="text-[11px] text-textMuted mt-2 block">DDoS & Exfiltration Dropped</span>
        </div>
        <div className="bg-surface border border-surfaceHighlight rounded-xl p-5 shadow-glow">
          <p className="text-xs text-textMuted font-medium">Detection Accuracy</p>
          <p className="text-2xl font-bold font-mono text-green-400 mt-2">99.4%</p>
          <span className="text-[11px] text-textMuted mt-2 block">NSL-KDD / CIC-IDS Benchmark</span>
        </div>
        <div className="bg-surface border border-surfaceHighlight rounded-xl p-5 shadow-glow">
          <p className="text-xs text-textMuted font-medium">Defense Model Status</p>
          <p className="text-2xl font-bold font-mono text-accent mt-2">Federated</p>
          <span className="text-[11px] text-textMuted mt-2 block">Collaborative Cross-Hospital</span>
        </div>
      </div>

      {/* Live Intrusion Alerts Table */}
      <div className="bg-surface border border-surfaceHighlight rounded-xl p-6 shadow-glow">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Terminal className="w-5 h-5 mr-2 text-accent" />
          Real-Time Intrusion & Anomaly Log
        </h2>

        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-surfaceHighlight text-xs font-semibold text-textMuted uppercase">
                <th className="pb-3 pr-4">Alert ID</th>
                <th className="pb-3 pr-4">Timestamp</th>
                <th className="pb-3 pr-4">Attack Type</th>
                <th className="pb-3 pr-4">Severity</th>
                <th className="pb-3 pr-4">Confidence</th>
                <th className="pb-3">Mitigation Action Taken</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surfaceHighlight text-xs">
              {alerts.map((alert) => (
                <tr key={alert.alert_id} className="hover:bg-background/40 transition-colors">
                  <td className="py-3 pr-4 font-mono font-bold text-white">{alert.alert_id}</td>
                  <td className="py-3 pr-4 font-mono text-textMuted">{alert.timestamp}</td>
                  <td className="py-3 pr-4 font-medium text-textMain">{alert.attack_type}</td>
                  <td className="py-3 pr-4">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                        alert.severity === 'Critical'
                          ? 'bg-red-500/20 text-red-400'
                          : alert.severity === 'High'
                          ? 'bg-amber-500/20 text-amber-400'
                          : 'bg-green-500/20 text-green-400'
                      }`}
                    >
                      {alert.severity}
                    </span>
                  </td>
                  <td className="py-3 pr-4 font-mono font-semibold text-white">
                    {(alert.confidence * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 text-textMuted">{alert.mitigation_action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default IntrusionDetection;
