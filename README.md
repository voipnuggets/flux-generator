# Flux Generator: macOS MLX-Powered Image Generation with Open WebUI compatable API
This repository is based on the Flux example from this repository.
https://github.com/ml-explore/mlx-examples/tree/main/flux

## Features

- Text-to-image generation
- Multiple model options (schnell/dev)
- Customizable image size and generation parameters
- Memory usage reporting
- API compatibility for third-party UIs like Open WebUI
- Unified server for both UI and API
- Configurable network access modes

## Screenshots:
![Flux App UI](flux_app_ui.jpg)

## Example Generation

Here's an example image generated using the Flux model:

![Moonset over ocean](generated_moonset.png)

Prompt: "a beautiful moonset over the ocean, highly detailed, 4k"
Parameters:
- Model: schnell
- Size: 512x512
- Steps: 2
- CFG Scale: 4.0

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+ (tested with python3.11)
- MLX framework

## Installation & Usage

### Quick Start (Recommended)

The easiest way to run Flux Generator is using the provided script:

```bash
# Make the script executable
chmod +x run_flux.sh

# Run in local-only mode (most secure)
./run_flux.sh

# Or run with network access (for remote access)
./run_flux.sh --network
```

The script will:
- Check if you're running on Apple Silicon Mac
- Create and set up a Python virtual environment
- Install all required dependencies
- Check for existing model files
- Start the server based on the selected mode

### Script Options

```bash
Usage: ./run_flux.sh [OPTIONS]

Options:
  -h, --help         Show this help message
  -n, --network      Enable network access (less secure)

Examples:
  ./run_flux.sh                 # Run in local-only mode (most secure)
  ./run_flux.sh --network       # Run with network access (for remote access)
```

### Access Modes

1. **Local Only (Default, Most Secure)**
   ```bash
   ./run_flux.sh
   ```
   - Only allows connections from localhost (127.0.0.1)
   - Best for local development and testing
   - Access via: http://127.0.0.1:7860

2. **Network Access**
   ```bash
   ./run_flux.sh --network
   ```
   - Allows connections from any network interface
   - Required for Docker integration
   - Less secure, use only in trusted networks
   - Access via:
     - Local: http://127.0.0.1:7860
     - Network: http://0.0.0.0:7860
     - Docker: http://host.docker.internal:7860

### Manual Installation

If you prefer to set things up manually:

1. **Create a virtual environment:**
   ```bash
   python3.11 -m venv venv
   
   # For bash/zsh:
   source venv/bin/activate
   
   # For fish:
   source venv/bin/activate.fish
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server:**
   ```bash
   # For local use only (most secure):
   python3.11 flux_app.py

   # For network access (remote):
   python3.11 flux_app.py --listen-all
   ```

### Command Line Options

```bash
python3.11 flux_app.py [OPTIONS]

Options:
  --port INTEGER       Port to run the server on (default: 7860)
  --listen-all        Listen on all network interfaces (0.0.0.0)
  --help              Show this message and exit
```

### Command Line Interface
For command-line image generation:

```bash
python3.11 txt2image.py --model schnell \
--n-images 1 \
--image-size 512x512 \
--verbose \
'A photo of an astronaut riding a horse on a beach.'
```

## Using the Web Interface

Once the server is running (either via `run_flux.sh` or manually):

1. Open your browser and navigate to http://127.0.0.1:7860
2. Enter a prompt and click the generate button
3. On first use, the model will be downloaded (approximately 23 GB)
4. Download progress will be visible in the terminal
5. Once downloaded, image generation will begin

## Generating image uising the flux generator UI

- The UI is accessable here http://127.0.0.1:7860
- Enter a prompt and click generate buttons
- On the first use the model will get downloaded which is about 23 GB in size
- The downloaded status is visible on the terminal
- Once the model is downloaded the image generation will start


## API Integration

The application provides an API that can be used with third-party UIs like Open WebUI.
Check this tutorial for Open WebUI integration instructions:
[Tutorial](https://voipnuggets.com/2025/02/18/flux-generator-local-image-generation-on-apple-silicon-with-open-webui-integration-using-flux-llm/
)

### Starting the Server

The server supports two access modes with different security levels:

1. Local Only (Most Secure):
   ```bash
   python3.11 flux_app.py
   ```
   - Only allows connections from localhost (127.0.0.1)
   - Best for local development and testing

2. All Networks:
   ```bash
   python3.11 flux_app.py --listen-all
   ```
   - Allows connections from any network interface
   - Less secure, use only in trusted networks

The server will start on port 7860 (configurable with `--port`).

### Integration with Open WebUI

Since Flux Generator requires direct access to Apple Silicon hardware, it runs natively on your Mac while Open WebUI can run in Docker:

1. Start Flux Generator with network access:
   ```bash
   ./run_flux.sh --network
   ```
   This will start the server and listen on all interfaces (required for Docker integration).

2. Run Open WebUI in Docker:
   ```bash
   docker run -d \
     -p 3000:8080 \
     --add-host=host.docker.internal:host-gateway \
     -e AUTOMATIC1111_BASE_URL=http://host.docker.internal:7860/ \
     -e ENABLE_IMAGE_GENERATION=True \
     -v open-webui:/app/backend/data \
     --name open-webui \
     --restart always \
     ghcr.io/open-webui/open-webui:main
   ```

3. Access Open WebUI at `http://localhost:3000`

The connection flow works like this:
```
Open WebUI (Docker Container) -> host.docker.internal:7860 -> Flux Generator (Native on Mac)
```

This setup ensures:
- Flux Generator has direct access to Apple Silicon for optimal performance
- Open WebUI runs in an isolated container
- Both services communicate seamlessly through Docker's networking

### Available Endpoints

1. `/sdapi/v1/txt2img` (POST)
   - Generate images from text
   - Parameters:
     ```json
     {
       "prompt": "your prompt here",
       "negative_prompt": "",
       "width": 512,
       "height": 512,
       "steps": 2,
       "cfg_scale": 4.0,
       "batch_size": 1,
       "n_iter": 1,
       "seed": -1,
       "model": "schnell"
     }
     ```

2. `/sdapi/v1/sd-models` (GET)
   - List available models
   - Returns Flux Schnell and Dev models

3. `/sdapi/v1/options` (GET/POST)
   - Get or set generation options
   - Includes model settings and parameters

4. `/sdapi/v1/progress` (GET)
   - Get generation progress information

### Example API Usage

Here's a Python example to generate images:

```python
import requests
import json
import base64

# Use appropriate URL based on your setup:
# Local only:      "http://127.0.0.1:7860"
url = "http://127.0.0.1:7860/sdapi/v1/txt2img"

payload = {
    "prompt": "a beautiful sunset over the ocean, highly detailed, 4k",
    "width": 512,
    "height": 512,
    "steps": 2,
    "cfg_scale": 4.0,
    "batch_size": 1,
    "n_iter": 1,
    "seed": 42,
    "model": "schnell"
}

response = requests.post(url, json=payload)
result = response.json()

# Save the generated image
if result["images"]:
    image_data = base64.b64decode(result["images"][0].split(",")[1])
    with open("generated_image.png", "wb") as f:
        f.write(image_data)
```

## Model Management

The Flux server requires model files to be downloaded before use. You can download the models in several ways:

1. Automatic download on first use:
   - Models will be downloaded automatically when you first try to generate an image
   - The download progress will be visible in the CLI/terminal
   - This may cause a delay on your first generation

2. Using HuggingFace CLI (Recommended for faster downloads):

   ```bash
   # Install the HuggingFace CLI
   pip install -U "huggingface_hub[cli]"

   # Install hf_transfer for blazingly fast speeds
   pip install hf_transfer

   # Login to your HF account
   huggingface-cli login
   
   # Enable fast downloads
   export HF_HUB_ENABLE_HF_TRANSFER=1

   # Download Schnell model
   huggingface-cli download black-forest-labs/FLUX.1-schnell

   # Download Dev model (optional)
   huggingface-cli download black-forest-labs/FLUX.1-dev
   ```

   Note: Each model is approximately 24GB in size. The download includes:
   - Model weights (flux1-{model}.safetensors)
   - Autoencoder (ae.safetensors)
   - Text encoders and tokenizers

3. Using the command-line interface:
   ```bash
   # This will download the model if not present locally
   python3.11 txt2image.py --model schnell --steps 1 --verbose "A photo of an astronaut riding a horse on Mars."
   ```
   - The download progress will be shown in the terminal
   - After download completes, it will generate the image

Model files are stored in the HuggingFace cache directory (`~/.cache/huggingface/hub/`).

## Buy Me a Coffee

👋 Hi, I'm Akash Gupta! Here's what I work on:

• 🚀 **Current Project**: Flux Generator - MLX-powered image generation for Apple Silicon
  - Local image generation using Apple's MLX framework
  - Beautiful Gradio UI with real-time stats
  - API compatible with Open WebUI
  - Memory-efficient design for M1/M2/M3 Macs

• 💼 **Professional Background**:
  - Sr. Voice Over IP Engineer
  - Expert in Kamailio and open-source VoIP
  - Cloud integration specialist

• 🌐 **Community Contributions**:
  - Blog: [voipnuggets.com](https://voipnuggets.com)
  - Focus: VoIP technology & AI advancements
  - Regular tutorials and technical guides

If you find this project helpful, consider supporting my work:

<img src="bmc_qr.png" alt="Buy Me A Coffee QR Code" width="200" height="200">

[☕ Buy Me a Coffee](https://buymeacoffee.com/akashg)

## Testing

The project includes several test suites to ensure everything works correctly:

### Shell Script Tests

To test the `run_flux.sh` script:
```bash
# Make the test script executable
chmod +x test/test_run_script.sh

# Run the tests
./test/test_run_script.sh
```

These tests verify:
- Command-line argument handling
- System requirement checks
- Virtual environment management
- Memory reporting
- Model file checking
- Network access modes

### Python Tests

To run the Python tests:
```bash
# Install test requirements
pip install -r test/requirements-test.txt

# Run all tests with coverage report
python3.11 test/run_tests.py
```

The Python tests cover:
- API endpoints and functionality
- UI components and handlers
- Model generation
- Docker integration
- Network connectivity

### Integration Tests

For testing Docker integration:
```bash
# Test connectivity between Flux and Open WebUI
python3.11 test/test_connectivity.py
```

This verifies:
- API accessibility
- Docker network configuration
- Model availability
- Generation capabilities

### Test Coverage

The test suite provides coverage reports for:
- Python code (via pytest-cov)
- API endpoints
- UI components
- Shell scripts

Coverage reports are generated in the `coverage_report` directory.