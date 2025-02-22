#!/bin/bash

# Set delimiter for sections
delimiter="----------------------------------------"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
YELLOW='\033[1;33m'

# Script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Parse command line arguments
LISTEN_ALL=0
HELP=0

print_usage() {
    printf "Usage: %s [OPTIONS]\n" "$0"
    printf "\nOptions:\n"
    printf "  -h, --help         Show this help message\n"
    printf "  -n, --network      Enable network access (less secure)\n"
    printf "\nExamples:\n"
    printf "  %s                 # Run in local-only mode (most secure)\n" "$0"
    printf "  %s --network       # Run with network access (for remote access)\n" "$0"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--network)
            LISTEN_ALL=1
            shift
            ;;
        -h|--help)
            HELP=1
            shift
            ;;
        *)
            printf "${RED}Unknown option: %s${NC}\n" "$1"
            print_usage
            exit 1
            ;;
    esac
done

if [ $HELP -eq 1 ]; then
    print_usage
    exit 0
fi

# Function to check if virtual environment is activated
check_venv() {
    if [ -z "$VIRTUAL_ENV" ]; then
        return 1
    fi
    
    if ! python -c "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)"; then
        return 1
    fi
    return 0
}

# Function to try activating virtual environment
try_activate_venv() {
    local activate_script="$1"
    if [ -f "$activate_script" ]; then
        printf "Trying activation script: %s\n" "$activate_script"
        source "$activate_script" 2>/dev/null
        if check_venv; then
            printf "${GREEN}Successfully activated virtual environment${NC}\n"
            return 0
        fi
    fi
    return 1
}

# Function to check system requirements
check_system() {
    printf "\n%s\n" "${delimiter}"
    printf "Checking system requirements...\n"
    
    # Check if running on macOS
    if [[ $(uname) != "Darwin" ]]; then
        printf "${RED}Error: This application requires macOS with Apple Silicon${NC}\n"
        return 1
    fi
    
    # Check if running on Apple Silicon
    if [[ $(uname -m) != "arm64" ]]; then
        printf "${RED}Error: This application requires Apple Silicon (M1/M2/M3)${NC}\n"
        return 1
    fi
    
    # Check Python version
    if ! command -v python3.11 &> /dev/null; then
        printf "${RED}Error: Python 3.11 is required but not found${NC}\n"
        return 1
    fi
    
    printf "${GREEN}System requirements met!${NC}\n"
    printf "\n%s\n" "${delimiter}"
    return 0
}

# Function to check and report memory
report_memory() {
    # Get total physical memory in GB
    total_mem=$(sysctl -n hw.memsize | awk '{printf "%.1f", $0/1024/1024/1024}')
    
    # Get VM stats
    vm_stat_output=$(vm_stat)
    
    # Calculate free memory more accurately
    pages_free=$(echo "$vm_stat_output" | awk '/Pages free/ {gsub(/\./, "", $3); print $3}')
    pages_inactive=$(echo "$vm_stat_output" | awk '/Pages inactive/ {gsub(/\./, "", $3); print $3}')
    pages_purgeable=$(echo "$vm_stat_output" | awk '/Pages purgeable/ {gsub(/\./, "", $3); print $3}')
    
    # Convert pages to GB (page size is 16384 bytes on Apple Silicon)
    page_size=16384
    free_mem=$(echo "scale=1; ($pages_free + $pages_inactive + $pages_purgeable) * $page_size / 1024 / 1024 / 1024" | bc)
    
    printf "\nSystem Memory:\n"
    printf "Total: %.1f GB\n" "$total_mem"
    printf "Available: %.1f GB\n" "$free_mem"
}

# Function to check model files
check_models() {
    printf "\n%s\n" "${delimiter}"
    printf "Checking model files...\n"
    
    MODEL_DIR="$HOME/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell"
    if [ -d "$MODEL_DIR" ]; then
        SIZE=$(du -sh "$MODEL_DIR" | cut -f1)
        # Check if size is at least 23GB
        SIZE_BYTES=$(du -s "$MODEL_DIR" | cut -f1)
        if [ "$SIZE_BYTES" -lt 24000000 ]; then  # Roughly 23GB in KB
            printf "${YELLOW}Warning: Model files found but may be incomplete (${SIZE})${NC}\n"
            printf "Expected size: ~23GB\n"
            printf "Found size: ${SIZE}\n"
            printf "Models will be re-downloaded if needed\n"
        else
            printf "${GREEN}Found existing model files (${SIZE})${NC}\n"
        fi
    else
        printf "${YELLOW}Model files will be downloaded on first use (~23GB required)${NC}\n"
    fi
    printf "\n%s\n" "${delimiter}"
}

# Main execution starts here
printf "\n%s\n" "${delimiter}"
printf "Starting Flux Generator Setup\n"
printf "\n%s\n" "${delimiter}"

# Check system requirements
check_system || exit 1

# Report memory status
report_memory

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    printf "\n%s\n" "${delimiter}"
    printf "${YELLOW}Creating virtual environment...${NC}\n"
    python3.11 -m venv venv
    first_launch=1
fi

# Activate virtual environment
printf "\n%s\n" "${delimiter}"
printf "${YELLOW}Activating virtual environment...${NC}\n"

# Try different activation scripts in sequence
VENV_PATH="$(pwd)/venv"
ACTIVATION_SUCCESSFUL=0

# First try bash/zsh activation (most common)
if try_activate_venv "$VENV_PATH/bin/activate"; then
    ACTIVATION_SUCCESSFUL=1
fi

# If bash activation failed, try fish
if [ $ACTIVATION_SUCCESSFUL -eq 0 ] && try_activate_venv "$VENV_PATH/bin/activate.fish"; then
    ACTIVATION_SUCCESSFUL=1
fi

# If both failed, try csh
if [ $ACTIVATION_SUCCESSFUL -eq 0 ] && try_activate_venv "$VENV_PATH/bin/activate.csh"; then
    ACTIVATION_SUCCESSFUL=1
fi

# If none worked, show error and exit
if [ $ACTIVATION_SUCCESSFUL -eq 0 ]; then
    printf "\n%s\n" "${delimiter}"
    printf "${RED}Error: Could not activate virtual environment${NC}\n"
    printf "Available activation scripts:\n"
    ls -l "$VENV_PATH/bin/activate"*
    printf "\nTry activating manually:\n"
    printf "For bash/zsh: source venv/bin/activate\n"
    printf "For fish: source venv/bin/activate.fish\n"
    printf "For csh: source venv/bin/activate.csh\n"
    printf "\n%s\n" "${delimiter}"
    exit 1
fi

# Install/upgrade pip first
printf "\n%s\n" "${delimiter}"
printf "Upgrading pip...\n"
python -m pip install --upgrade pip

# Install requirements
printf "\n%s\n" "${delimiter}"
printf "Installing requirements...\n"
pip install -r requirements.txt

# Check model files
check_models

# Run the application
printf "\n%s\n" "${delimiter}"
printf "${GREEN}Starting Flux Generator...${NC}\n"

if [ $LISTEN_ALL -eq 1 ]; then
    printf "${YELLOW}Warning: Server will listen on all network interfaces (0.0.0.0)${NC}\n"
    printf "This mode is less secure and should only be used in trusted networks\n\n"
    printf "Server access:\n"
    printf "  Local:    http://127.0.0.1:7860\n"
    printf "  Network:  http://0.0.0.0:7860\n"
    printf "  Docker:   http://host.docker.internal:7860\n"
    
    # Start server with network access
    python flux_app.py --listen-all --port 7860
else
    printf "Running in local-only mode (most secure)\n\n"
    printf "Server access:\n"
    printf "  Local:    http://127.0.0.1:7860\n"
    
    # Start server in local-only mode
    python flux_app.py --port 7860
fi

printf "\n%s\n" "${delimiter}" 