#!/bin/bash
# AIGUIBook - First Time Setup Script
# Run this on your machine with RTX 5080

set -e

GREEN="${GREEN}\033[0;32m}"
YELLOW="${YELLOW}\033[1;33m}"
CYAN="${CYAN}\033[0;36m}"
NC="${NC}\033[0m}"

echo "${CYAN}============================================${NC}"
echo "${CYAN}  AIGUIBook - Setup Script${NC}"
echo "${CYAN}============================================${NC}"
echo ""

# Check Python version
echo "${YELLOW}[1/7] Checking Python...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP "[0-9]+\.[0-9]+" || echo "0.0")
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "  Warning: Python 3.10+ recommended (found $PYTHON_VERSION)"
    echo "  Install with: sudo apt install python3.12 python3.12-venv python3.12-dev"
else
    echo "  OK: Python $PYTHON_VERSION"
fi

# Check GPU
echo "${YELLOW}[2/7] Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "  GPU detected!"
else
    echo "  Warning: nvidia-smi not found. Is NVIDIA driver installed?"
fi

# Check FFmpeg
echo "${YELLOW}[3/7] Checking FFmpeg...${NC}"
if command -v ffmpeg &> /dev/null; then
    echo "  OK: $(ffmpeg -version | head -1)"
else
    echo "  Installing FFmpeg..."
    sudo apt install -y ffmpeg
fi

# Create virtual environment
echo "${YELLOW}[4/7] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Created venv/"
else
    echo "  venv/ already exists"
fi

source venv/bin/activate

# Install PyTorch
echo "${YELLOW}[5/7] Installing PyTorch with CUDA 12.4...${NC}"
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "${YELLOW}[6/7] Installing project dependencies...${NC}"
pip install -r requirements.txt

# Install qwen-tts
echo "${YELLOW}[6.5/7] Installing Qwen3-TTS...${NC}"
pip install qwen-tts

# Optional: flash attention
echo "${YELLOW}[6.6/7] (Optional) Flash attention...${NC}"
echo "  Skipping flash-attn (optional, can install later with: pip install flash-attn --no-build-isolation)"

# Create directories
echo "${YELLOW}[7/7] Creating project directories...${NC}"
mkdir -p voices output work epub
mkdir -p ~/.aiguibook

echo ""

# Create initial config
echo "${CYAN}============================================${NC}"
echo "${CYAN}  Setup Complete!${NC}"
echo "${CYAN}============================================${NC}"
echo ""
echo "  1. Make sure LM Studio is running with Gemma 4 26B A4B loaded"
echo "     (default port: 1234)"
echo ""
echo "  2. Start the GUI:"
echo "     ${GREEN}source venv/bin/activate${NC}"
echo "     ${GREEN}python main.py${NC}"
echo ""
echo "  3. Or use the CLI:"
echo "     ${GREEN}python cli.py generate --input my_book.epub --output ./output/${NC}"
echo ""
echo "  GUI will be available at: ${GREEN}http://localhost:7860${NC}"
echo ""
echo "  Config file: ${GREEN}~/.aiguibook/config.yaml${NC}"
echo ""
