import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ShieldAlert, Lock, User, KeyRound, ArrowRight } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';

export const Login: React.FC = () => {
  const [username, setUsername] = useState('dr_smith');
  const [password, setPassword] = useState('password123');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    try {
      // Direct demo login
      await login('demo_jwt_token_fedmedshield_2026');
      navigate('/dashboard');
    } catch {
      setError('Invalid clinical credentials.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-textMain flex items-center justify-center p-6 relative overflow-hidden">
      {/* Background Glow */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-accent/20 rounded-full blur-3xl pointer-events-none" />

      <div className="w-full max-w-md bg-surface border border-surfaceHighlight rounded-2xl p-8 shadow-2xl relative z-10 backdrop-blur-md">
        <div className="text-center mb-8">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-accent to-primary flex items-center justify-center text-white mx-auto shadow-glow mb-4">
            <ShieldAlert className="w-7 h-7" />
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">FedMedShield Portal</h1>
          <p className="text-sm text-textMuted mt-1">Multi-Task Federated Healthcare Security</p>
        </div>

        {error && (
          <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs">
            {error}
          </div>
        )}

        <form onSubmit={handleLogin} className="space-y-5">
          <div>
            <label className="block text-xs font-semibold text-textMuted uppercase tracking-wider mb-2">
              Investigator Username
            </label>
            <div className="relative">
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full bg-background border border-surfaceHighlight rounded-xl pl-10 pr-4 py-2.5 text-sm text-textMain focus:outline-none focus:border-accent transition-colors"
                required
              />
              <User className="w-4 h-4 text-textMuted absolute left-3.5 top-1/2 -translate-y-1/2" />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-textMuted uppercase tracking-wider mb-2">
              Cryptographic Password
            </label>
            <div className="relative">
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-background border border-surfaceHighlight rounded-xl pl-10 pr-4 py-2.5 text-sm text-textMain focus:outline-none focus:border-accent transition-colors"
                required
              />
              <KeyRound className="w-4 h-4 text-textMuted absolute left-3.5 top-1/2 -translate-y-1/2" />
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 bg-gradient-to-r from-accent to-primary text-white font-semibold rounded-xl shadow-lg hover:shadow-accent/30 transition-all flex items-center justify-center space-x-2 mt-4"
          >
            <span>{isLoading ? 'Authenticating...' : 'Sign In to Clinical Silo'}</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </form>

        <div className="mt-6 pt-6 border-t border-surfaceHighlight text-center text-xs text-textMuted">
          <span>Protected by Elliptic Curve Diffie-Hellman & RDP Guard</span>
        </div>
      </div>
    </div>
  );
};

export default Login;
