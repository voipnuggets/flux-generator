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
uvicorn flux_app:app --reload
```

## Features

- Text-to-image generation
- Multiple model options (schnell/dev)
- Customizable image size and generation parameters
- Memory usage reporting