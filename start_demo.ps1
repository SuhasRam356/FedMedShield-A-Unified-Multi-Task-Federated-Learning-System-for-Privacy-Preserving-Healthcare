<#
.SYNOPSIS
    Starts the FedMedShield project using PM2.
.DESCRIPTION
    This script ensures the logs directory exists and boots the PM2 ecosystem
    for the FastAPI backend and Vite frontend.
#>

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " Booting FedMedShield - IEEE Federated Learning System" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

# Create logs directory if it doesn't exist
if (-not (Test-Path -Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
    Write-Host "[+] Created logs directory." -ForegroundColor Green
}

# Check if pm2 is installed globally
$pm2Installed = Get-Command pm2 -ErrorAction SilentlyContinue
if (-not $pm2Installed) {
    Write-Host "[!] PM2 is not installed. Installing globally via npm..." -ForegroundColor Yellow
    npm install -g pm2
}

# Start the ecosystem (Backend)
Write-Host "[+] Starting PM2 Ecosystem (Backend)..." -ForegroundColor Green
pm2 start ecosystem.config.js

# Start Frontend natively
Write-Host "[+] Starting Vite Frontend (Native)..." -ForegroundColor Green
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit -Command `"cd frontend; npm run dev`""

Write-Host ""
Write-Host "System is now running!" -ForegroundColor Green
Write-Host "-> Dashboard: http://localhost:5173" -ForegroundColor Blue
Write-Host "-> Backend API: http://localhost:8000" -ForegroundColor Blue
Write-Host ""
Write-Host "Useful Commands:" -ForegroundColor Yellow
Write-Host "  pm2 status    - View backend status"
Write-Host "  pm2 logs      - View backend logs"
Write-Host "  pm2 stop all  - Stop the backend"
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
