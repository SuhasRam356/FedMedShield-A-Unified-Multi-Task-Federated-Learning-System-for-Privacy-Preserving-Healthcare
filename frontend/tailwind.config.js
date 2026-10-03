/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0B1120',
        card: '#1E293B',
        primary: '#3B82F6',
        secondary: '#10B981',
        accent: '#8B5CF6',
        textMain: '#F8FAFC',
        textMuted: '#94A3B8'
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
