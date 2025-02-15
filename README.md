# FLUX Image Generator

A Python application for generating images using the FLUX model on Apple Silicon Macs.

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

## Installation

pip install -r requirements.txt

## Usage

Run the Gradio interface:
```bash
python gradio_app.py
```

Run the FastAPI server:
```bash
# For local use only (most secure):
python3.11 flux_app.py --enable-api

# For local network access (LAN only):
python3.11 flux_app.py --enable-api --listen-local

# For all network access (including Docker):
python3.11 flux_app.py --enable-api --listen-all
```

### Command Line Interface
Generate images using the command line:

```bash
python3.11 txt2image.py --model schnell \
--n-images 1 \
--image-size 512x512 \
--verbose \
'A photo of an astronaut riding a horse on a beach.'
```

## Features

- Text-to-image generation
- Multiple model options (schnell/dev)
- Customizable image size and generation parameters
- Memory usage reporting
- Stable Diffusion API compatibility for third-party UIs
- Configurable network access modes

## API Integration

The application provides a Stable Diffusion-compatible API that can be used with third-party UIs like Open WebUI.

### Starting the API Server

The server supports three access modes with different security levels:

1. Local Only (Most Secure):
   ```bash
   python3.11 flux_app.py --enable-api
   ```
   - Only allows connections from localhost (127.0.0.1)
   - Best for local development and testing
   - Not accessible from Docker or other machines

2. Local Network:
   ```bash
   python3.11 flux_app.py --enable-api --listen-local
   ```
   - Allows connections from your local network (LAN)
   - Good for accessing from other devices on your network
   - More secure than listening on all interfaces

3. All Networks:
   ```bash
   python3.11 flux_app.py --enable-api --listen-all
   ```
   - Allows connections from any network interface
   - Required for Docker integration
   - Less secure, use only in trusted networks

The server will start on port 7860 (configurable with `--api-port`).

### Docker Integration with Open WebUI

1. Start the Flux API server on your host machine:
   ```bash
   python3.11 flux_app.py --enable-api --listen-all
   ```
   Note: Docker integration requires `--listen-all` to allow container access.

2. Run Open WebUI in Docker:
   ```bash
   docker run -d -p 3000:8080 \
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
Open WebUI (Docker Container) -> host.docker.internal:7860 -> Flux API (Host Machine)
```

This setup runs the resource-intensive model natively on your Mac while the UI runs in Docker.

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
# Local network:   "http://192.168.1.1:7860"
# Docker:          "http://host.docker.internal:7860"
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

## Notes

- The API is designed to be compatible with Stable Diffusion Web UI's API format
- Default port is 7860, can be changed with `--api-port`
- Three network access modes available:
  - `--enable-api`: Local only (most secure)
  - `--listen-local`: Local network access
  - `--listen-all`: All network access (required for Docker)
- CORS is enabled to allow requests from web UIs
- The schnell model uses 2 steps by default, while the dev model uses 50 steps

# Model Management

The Flux API server requires model files to be downloaded before use. You can download the models in two ways:

1. Automatic download on first use:
   - Models will be downloaded automatically when you first try to generate an image
   - This may cause a delay on your first generation

2. Manual download with flags:
   ```bash
   # Download models if not present
   python3.11 flux_app.py --download-models
   
   # Force re-download models (overwrites existing files)
   python3.11 flux_app.py --force-download
   ```

Model files are stored in `~/.flux/models/` directory.

# Running the Server

You can run the Flux server in different modes depending on your needs:

1. Local Only (Most Secure):
   ```bash
   python3.11 flux_app.py --enable-api
   ```
   - Only allows connections from localhost
   - Best for local development and testing

2. Local Network:
   ```bash
   python3.11 flux_app.py --enable-api --listen-local
   ```
   - Allows connections from your local network (192.168.0.0/16, 10.0.0.0/8)
   - Useful for accessing from other devices on your network

3. All Networks:
   ```bash
   python3.11 flux_app.py --enable-api --listen-all
   ```
   - Allows connections from any IP address
   - Required for Docker container access
   - Use with caution, only on trusted networks

Additional options:
- `--api-port PORT`: Change the server port (default: 7860)
- `--download-models`: Download required models if not present
- `--force-download`: Force re-download of models