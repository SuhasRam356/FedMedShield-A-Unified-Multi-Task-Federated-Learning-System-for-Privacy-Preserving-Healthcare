const { spawn } = require('child_process');
const path = require('path');

console.log("Starting Vite Frontend Wrapper...");

// Spawn npm run dev in the frontend folder
// We use shell: true to properly execute npm.cmd on Windows
const frontend = spawn('npm', ['run', 'dev'], {
  cwd: path.join(__dirname, 'frontend'),
  shell: true,
  stdio: 'inherit'
});

frontend.on('close', (code) => {
  console.log(`Frontend exited with code ${code}`);
  process.exit(code);
});

frontend.on('error', (err) => {
  console.error("Failed to start frontend process:", err);
  process.exit(1);
});
