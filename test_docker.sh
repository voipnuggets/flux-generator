#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
YELLOW='\033[1;33m'

echo -e "${YELLOW}Testing Flux Generator Docker Setup${NC}"
echo "=================================="

# Step 1: Check Docker installation
echo -e "\n1. Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker is installed${NC}"

# Step 2: Check Docker Compose installation
echo -e "\n2. Checking Docker Compose..."
if ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed or not working.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker Compose is installed${NC}"

# Step 3: Check for existing model files
echo -e "\n3. Checking for existing model files..."
HF_CACHE_DIR="$HOME/.cache/huggingface/hub"
if [ -d "$HF_CACHE_DIR" ]; then
    if ls $HF_CACHE_DIR/*/FLUX.1-schnell/* &> /dev/null; then
        echo -e "${GREEN}✅ Found existing model files in HuggingFace cache${NC}"
        echo -e "   Cache directory: $HF_CACHE_DIR"
    else
        echo -e "${YELLOW}⚠️  No existing model files found. They will be downloaded during first generation.${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  HuggingFace cache directory not found. Models will be downloaded during first generation.${NC}"
fi

# Step 4: Build the Docker image
echo -e "\n4. Building Docker image..."
if ! docker compose build; then
    echo -e "${RED}❌ Failed to build Docker image${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker image built successfully${NC}"

# Step 5: Start the container
echo -e "\n5. Starting container..."
if ! docker compose up -d; then
    echo -e "${RED}❌ Failed to start container${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Container started successfully${NC}"

# Step 6: Verify HuggingFace cache mount
echo -e "\n6. Verifying HuggingFace cache mount..."
if ! docker compose exec flux-generator ls /root/.cache/huggingface/hub &> /dev/null; then
    echo -e "${RED}❌ HuggingFace cache not properly mounted${NC}"
    docker compose logs
    docker compose down
    exit 1
fi
echo -e "${GREEN}✅ HuggingFace cache mounted successfully${NC}"

# Step 7: Wait for service to be ready
echo -e "\n7. Waiting for service to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:7860 > /dev/null; then
        echo -e "${GREEN}✅ Service is up and running${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Service failed to start within 30 seconds${NC}"
        docker compose logs
        docker compose down
        exit 1
    fi
    echo -n "."
    sleep 1
done

# Step 8: Test API endpoints
echo -e "\n8. Testing API endpoints..."

# Test models endpoint
echo -n "   Testing /sdapi/v1/sd-models endpoint... "
if curl -s http://localhost:7860/sdapi/v1/sd-models | grep -q "flux-schnell"; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
    docker compose logs
    docker compose down
    exit 1
fi

# Test options endpoint
echo -n "   Testing /sdapi/v1/options endpoint... "
if curl -s http://localhost:7860/sdapi/v1/options | grep -q "sd_model_checkpoint"; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
    docker compose logs
    docker compose down
    exit 1
fi

# Step 9: Test image generation
echo -e "\n9. Testing image generation..."
echo "   Using existing model files from HuggingFace cache..."

TEST_PAYLOAD='{
    "prompt": "sun rise over a calm ocean with a few clouds",
    "width": 512,
    "height": 512,
    "steps": 1,
    "cfg_scale": 4.0,
    "batch_size": 1,
    "n_iter": 1,
    "seed": 42,
    "model": "schnell"
}'

if curl -s -X POST http://localhost:7860/sdapi/v1/txt2img \
    -H "Content-Type: application/json" \
    -d "$TEST_PAYLOAD" \
    | grep -q "images"; then
    echo -e "${GREEN}✅ Image generation successful${NC}"
else
    echo -e "${RED}❌ Image generation failed${NC}"
    docker compose logs
    docker compose down
    exit 1
fi

# Step 10: Clean up
echo -e "\n10. Cleaning up..."
docker compose down
echo -e "${GREEN}✅ Cleanup complete${NC}"

echo -e "\n${GREEN}All tests passed successfully!${NC}"
echo -e "\nTo start the service again, run: ${YELLOW}docker compose up -d${NC}" 