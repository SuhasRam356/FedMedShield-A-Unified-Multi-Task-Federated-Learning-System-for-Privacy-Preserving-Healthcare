#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# FedMedShield — Auto Setup Script
# Run: bash setup.sh
# ═══════════════════════════════════════════════════════════════

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    FedMedShield — Automated Setup                 ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"

# ── Step 1: Python virtual environment ──
echo -e "\n${YELLOW}[1/5]${NC} Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo -e "${GREEN}  ✓ venv created${NC}"
else
    echo -e "${GREEN}  ✓ venv already exists${NC}"
fi

# Activate venv
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# ── Step 2: Install Python dependencies ──
echo -e "\n${YELLOW}[2/5]${NC} Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r backend/requirements.txt -q
echo -e "${GREEN}  ✓ Python packages installed${NC}"

# ── Step 3: Install frontend dependencies ──
echo -e "\n${YELLOW}[3/5]${NC} Installing frontend dependencies..."
cd frontend
npm install
cd ..
echo -e "${GREEN}  ✓ Frontend packages installed${NC}"

# ── Step 4: Install PM2 globally ──
echo -e "\n${YELLOW}[4/5]${NC} Installing PM2 process manager..."
npm install -g pm2 2>/dev/null || echo "  ⚠ PM2 install requires sudo — run: sudo npm install -g pm2"
echo -e "${GREEN}  ✓ PM2 ready${NC}"

# ── Step 5: Create .env if missing ──
echo -e "\n${YELLOW}[5/5]${NC} Checking .env file..."
if [ ! -f ".env" ]; then
    cp .env.example .env 2>/dev/null || echo "  .env already present"
fi
echo -e "${GREEN}  ✓ Environment configured${NC}"

# ── Done ──
echo -e "\n${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ FedMedShield setup complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${YELLOW}To start everything:${NC}"
echo -e "    python run_all.py"
echo -e "    ${YELLOW}OR${NC}"
echo -e "    pm2 start ecosystem.config.js"
echo ""
echo -e "  ${YELLOW}Access:${NC}"
echo -e "    🎨 Frontend   → http://localhost:5173"
echo -e "    ⚙️  API Docs   → http://localhost:8000/docs"
echo -e "    🌐 FL Server  → localhost:8080"
echo -e "    📊 PM2 Monitor → pm2 monit"
echo ""
