#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
YELLOW='\033[1;33m'

# Check if running on macOS
if [[ $(uname) != "Darwin" ]]; then
    echo -e "${RED}Error: This application requires macOS with Apple Silicon${NC}"
    exit 1
fi

# Check if running on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo -e "${RED}Error: This application requires Apple Silicon (M1/M2/M3)${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3.11 -m venv venv
fi

# Activate virtual environment based on shell
SHELL_NAME=$(basename "$SHELL")
case "$SHELL_NAME" in
    "fish")
        source venv/bin/activate.fish
        ;;
    "zsh"|"bash")
        source venv/bin/activate
        ;;
    *)
        echo -e "${RED}Unsupported shell: $SHELL_NAME${NC}"
        echo -e "${YELLOW}Please activate the virtual environment manually:${NC}"
        echo -e "For bash/zsh: source venv/bin/activate"
        echo -e "For fish: source venv/bin/activate.fish"
        exit 1
        ;;
esac

# Install requirements
echo -e "${YELLOW}Installing requirements...${NC}"
pip install -r requirements.txt

# Check for model files
MODEL_DIR="$HOME/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell"
if [ -d "$MODEL_DIR" ]; then
    SIZE=$(du -sh "$MODEL_DIR" | cut -f1)
    echo -e "${GREEN}Found existing model files (${SIZE})${NC}"
else
    echo -e "${YELLOW}Model files will be downloaded on first use${NC}"
fi

# Run the application
echo -e "${GREEN}Starting Flux Generator...${NC}"
python3.11 flux_app.py --listen-all --port 7860 