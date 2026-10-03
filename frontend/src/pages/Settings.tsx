import React from 'react';
import { Settings as SettingsIcon, Bell, Sliders, Globe } from 'lucide-react';

const Settings: React.FC = () => {
  return (
    <div className="space-y-6 fade-in">
      <div>
        <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-200 to-gray-500">
          Global Settings
        </h1>
        <p className="text-textMuted mt-1">
          Configure overarching system preferences and network parameters.
        </p>
        <div className="mt-3 p-3 bg-card border border-white/10 rounded-lg text-xs text-textMuted flex items-center justify-between">
          <span>ℹ️ Prototype Notice: Settings configured in this view represent local presentation defaults for UI demonstration.</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        
        {/* Settings Navigation (Simulated) */}
        <div className="md:col-span-1 space-y-2">
          <div className="p-3 bg-surfaceHighlight/50 border-l-4 border-accent text-textMain rounded-r font-medium flex items-center cursor-pointer">
            <Sliders className="w-5 h-5 mr-3 text-accent" />
            General
          </div>
          <div className="p-3 text-textMuted hover:bg-surface hover:text-textMain rounded font-medium flex items-center transition-colors cursor-pointer">
            <Bell className="w-5 h-5 mr-3" />
            Alerts & Notifications
          </div>
          <div className="p-3 text-textMuted hover:bg-surface hover:text-textMain rounded font-medium flex items-center transition-colors cursor-pointer">
            <Globe className="w-5 h-5 mr-3" />
            API Integrations
          </div>
        </div>

        {/* Settings Content */}
        <div className="md:col-span-2 space-y-6">
          <div className="bg-surface border border-surfaceHighlight rounded-xl p-6 shadow-glow">
            <h2 className="text-xl font-semibold flex items-center mb-6">
              <SettingsIcon className="mr-3 text-gray-400" />
              General Configuration
            </h2>

            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-textMuted mb-2">Federation Name</label>
                <input 
                  type="text" 
                  defaultValue="FedMedShield Multi-Task Cohort A" 
                  className="w-full bg-background border border-surfaceHighlight rounded-lg px-4 py-2 text-textMain focus:outline-none focus:border-accent transition-colors"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-textMuted mb-2">Central Server Bind Address</label>
                <input 
                  type="text" 
                  defaultValue="0.0.0.0:8000" 
                  className="w-full bg-background border border-surfaceHighlight rounded-lg px-4 py-2 text-textMain focus:outline-none focus:border-accent transition-colors font-mono text-sm"
                />
              </div>

              <div className="flex items-center justify-between py-2 border-b border-surfaceHighlight">
                <div>
                  <h4 className="text-sm font-medium text-textMain">Enforce Strict HTTPS</h4>
                  <p className="text-xs text-textMuted mt-1">Require TLS 1.3 for all inter-node communication.</p>
                </div>
                <div className="w-11 h-6 bg-accent rounded-full relative cursor-pointer">
                  <div className="w-5 h-5 bg-white rounded-full absolute right-0.5 top-0.5 shadow-sm"></div>
                </div>
              </div>

              <div className="flex items-center justify-between py-2 border-b border-surfaceHighlight">
                <div>
                  <h4 className="text-sm font-medium text-textMain">Audit Logging</h4>
                  <p className="text-xs text-textMuted mt-1">Log all cryptographic handshakes and weight aggregations.</p>
                </div>
                <div className="w-11 h-6 bg-accent rounded-full relative cursor-pointer">
                  <div className="w-5 h-5 bg-white rounded-full absolute right-0.5 top-0.5 shadow-sm"></div>
                </div>
              </div>

              <div className="pt-4 flex justify-end">
                <button className="px-6 py-2 bg-gradient-to-r from-accent to-accentHover text-white font-medium rounded-lg shadow-lg hover:shadow-accent/25 transition-all">
                  Save Changes
                </button>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default Settings;
