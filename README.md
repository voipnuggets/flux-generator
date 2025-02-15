# FLUX Image Generator

A Python application for generating images using the FLUX model on Apple Silicon Macs.

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
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
python3.11 flux_app.py --enable-api
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

## API Integration

The application provides a Stable Diffusion-compatible API that can be used with third-party UIs like Open WebUI.

### Starting the API Server

```bash
python3.11 flux_app.py --enable-api
```

This will start the server on `http://127.0.0.1:7860`.

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

### Open WebUI Integration

1. Start the Flux API server:
   ```bash
   python3.11 flux_app.py --enable-api
   ```

2. In Open WebUI:
   - Go to Settings > API Connections
   - Add a new connection with URL: `http://127.0.0.1:7860`
   - The Flux models will appear in the model selection dropdown

### Example API Usage

Here's a Python example to generate images:

```python
import requests
import json
import base64

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
- CORS is enabled to allow requests from web UIs
- The schnell model uses 2 steps by default, while the dev model uses 50 steps