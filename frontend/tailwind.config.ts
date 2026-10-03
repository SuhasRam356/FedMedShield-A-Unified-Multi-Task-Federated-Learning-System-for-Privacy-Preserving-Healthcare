// ═══════════════════════════════════════════════════════════════
// FedMedShield — TailwindCSS Configuration
// Dark medical theme: #0d0f1a background, #60a5fa accent
// ═══════════════════════════════════════════════════════════════

import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Core dark medical palette
        med: {
          bg: '#0d0f1a',
          surface: '#141729',
          card: '#1a1e35',
          border: '#2a2f4a',
          accent: '#60a5fa',
          'accent-bright': '#93c5fd',
          success: '#34d399',
          warning: '#fbbf24',
          danger: '#f87171',
          critical: '#ef4444',
          text: '#e2e8f0',
          muted: '#94a3b8',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(96, 165, 250, 0.3)' },
          '100%': { boxShadow: '0 0 20px rgba(96, 165, 250, 0.6)' },
        },
      },
    },
  },
  plugins: [],
};

export default config;
