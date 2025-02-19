import io
import base64
import argparse
import gradio as gr
from typing import Optional, List, Tuple, Union
import mlx.core as mx
from PIL import Image
import numpy as np
import platform
import sys
from pathlib import Path
import webbrowser
from flux import FluxPipeline
from flux.utils import configs, hf_hub_download
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json

# Global pipeline instances
flux_pipeline = None

# API Models
class SDAPIRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    steps: Optional[int] = None
    cfg_scale: float = 4.0
    batch_size: int = 1
    n_iter: int = 1
    seed: int = -1
    model: str = "schnell"

class SDAPIResponse(BaseModel):
    images: List[str]
    parameters: dict
    info: str

class FluxAPI:
    """Unified API for both UI and external API calls"""
    def __init__(self):
        self.pipeline = None
        self.current_model = None
    
    def init_pipeline(self, model: str):
        """Initialize the pipeline if needed"""
        if self.pipeline is None or self.current_model != model:
            model_name = "flux-" + model
            self.pipeline = FluxPipeline(model_name)
            self.current_model = model
        return self.pipeline
    
    async def txt2img(self, request: SDAPIRequest) -> SDAPIResponse:
        """Generate images from text"""
        try:
            images = self.generate_images(
                prompt=request.prompt,
                model=request.model,
                width=request.width,
                height=request.height,
                steps=request.steps,
                guidance=request.cfg_scale,
                seed=request.seed if request.seed >= 0 else None,
                batch_size=request.batch_size,
                n_iter=request.n_iter,
                return_pil=False  # Get base64 images for API
            )
            
            return SDAPIResponse(
                images=images,
                parameters={
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "width": request.width,
                    "height": request.height,
                    "steps": request.steps,
                    "cfg_scale": request.cfg_scale,
                    "seed": request.seed,
                    "model": request.model
                },
                info=f"Generated with Flux {request.model} model"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def generate_images(
        self,
        prompt: str,
        model: str = "schnell",
        width: int = 512,
        height: int = 512,
        steps: Optional[int] = None,
        guidance: float = 4.0,
        seed: Optional[int] = None,
        batch_size: int = 1,
        n_iter: int = 1,
        return_pil: bool = False
    ) -> List[Union[str, Image.Image]]:
        """Generate images with the given parameters"""
        # Initialize pipeline
        pipeline = self.init_pipeline(model)
        
        # Parse image size
        latent_size = (height // 8, width // 8)
        steps = steps or (50 if model == "dev" else 2)
        
        # Generate latents
        latents = pipeline.generate_latents(
            prompt,
            n_images=batch_size * n_iter,
            num_steps=steps,
            latent_size=latent_size,
            guidance=guidance,
            seed=seed
        )
        
        # Process latents
        conditioning = next(latents)
        mx.eval(conditioning)
        
        for x_t in latents:
            mx.eval(x_t)
        
        # Decode images
        decoded = []
        for i in range(0, batch_size * n_iter):
            decoded.append(pipeline.decode(x_t[i:i+1], latent_size))
            mx.eval(decoded[-1])
        
        # Convert to PIL Images or base64
        images = []
        for img_tensor in decoded:
            img_array = (mx.array(img_tensor[0]) * 255).astype(mx.uint8)
            pil_image = Image.fromarray(np.array(img_array))
            
            if return_pil:
                images.append(pil_image)
            else:
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                images.append(f"data:image/png;base64,{img_str}")
        
        return images
    
    def list_models(self):
        """List available models"""
        return [
            {
                "title": "Flux Schnell",
                "model_name": "flux-schnell",
                "filename": "flux-schnell"
            },
            {
                "title": "Flux Dev",
                "model_name": "flux-dev",
                "filename": "flux-dev"
            }
        ]
    
    def get_options(self):
        """Get current options"""
        return {
            "sd_model_checkpoint": "flux-schnell",
            "sd_backend": "Flux MLX"
        }
    
    def set_options(self, options: dict):
        """Set options"""
        return {"success": True}
    
    def get_progress(self):
        """Get generation progress"""
        return {
            "progress": 0,
            "eta_relative": 0,
            "state": {
                "skipped": False,
                "interrupted": False,
                "job": "",
                "job_count": 0,
                "job_timestamp": ""
            },
            "current_image": None,
            "textinfo": "Idle"
        }

# Create global API instance
api = FluxAPI()

def create_api(app):
    """Create and mount API endpoints to the FastAPI app"""
    @app.post("/sdapi/v1/txt2img")
    async def txt2img(request: SDAPIRequest):
        return await api.txt2img(request)

    @app.get("/sdapi/v1/sd-models")
    async def list_models():
        return api.list_models()

    @app.get("/sdapi/v1/options")
    async def get_options():
        return api.get_options()

    @app.post("/sdapi/v1/options")
    async def set_options(options: dict):
        return api.set_options(options)

    @app.get("/sdapi/v1/progress")
    async def get_progress():
        return api.get_progress()

    return api

def check_system_compatibility():
    """Check if running on Apple Silicon Mac"""
    if sys.platform != 'darwin':
        raise SystemError("This application only runs on macOS")
    
    if platform.machine() != 'arm64':
        raise SystemError("This application requires an Apple Silicon Mac")
    
    return True

def to_latent_size(size: Tuple[int, int]) -> Tuple[int, int]:
    """Convert image size to latent size"""
    h, w = size
    h = ((h + 15) // 16) * 16
    w = ((w + 15) // 16) * 16

    if (h, w) != size:
        print(
            "Warning: The image dimensions need to be divisible by 16px. "
            f"Changing size to {h}x{w}."
        )

    return (h // 8, w // 8)

def check_port_available(host: str, port: int) -> bool:
    """Check if a port is available on the given host"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False

def find_available_port(host: str, start_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if check_port_available(host, port):
            return port
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts - 1}")

def create_ui():
    """Create the Gradio interface"""
    css = """
        .container { max-width: 1200px; margin: auto; }
        .prompt-box { min-height: 100px; }
        .status-box { margin-top: 10px; }
        .generate-btn { min-height: 60px; }
    """
    
    blocks = gr.Blocks(theme=gr.themes.Soft(), css=css)
    with blocks:
        # Main Content
        with gr.Column(elem_classes="container"):
            gr.Markdown(
                """
                # 🎨 FLUX Image Generator
                Generate beautiful images from text using the FLUX model on Apple Silicon Macs.
                """
            )
            
            with gr.Row():
                # Left Column - Controls
                with gr.Column(scale=1):
                    # Generation Controls
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here... (e.g., 'a beautiful moonset over the ocean, highly detailed, 4k')",
                        lines=3,
                        elem_classes="prompt-box"
                    )
                    
                    with gr.Group():
                        gr.Markdown("### ⚙️ Model Settings")
                        model_type = gr.Radio(
                            choices=["schnell", "dev"],
                            value="schnell",
                            label="Model Type",
                            info="Schnell (2 steps) is faster, Dev (50 steps) is higher quality"
                        )
                        
                        with gr.Row():
                            num_steps = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=None,
                                label="Steps",
                                info="Leave empty for default (2 for Schnell, 50 for Dev)"
                            )
                            guidance = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                step=0.5,
                                value=4.0,
                                label="Guidance Scale",
                                info="Higher values = stronger adherence to prompt"
                            )
                    
                    # Add model-specific parameter updates
                    def update_model_params(model):
                        if model == "dev":
                            return {
                                num_steps: gr.update(value=50, minimum=1, maximum=100, interactive=True),
                                guidance: gr.update(value=4.0, interactive=True)
                            }
                        else:  # schnell
                            return {
                                num_steps: gr.update(value=2, minimum=1, maximum=100, interactive=True),
                                guidance: gr.update(value=4.0, interactive=True)
                            }
                    
                    # Connect the model selection to parameter updates
                    model_type.change(
                        fn=update_model_params,
                        inputs=[model_type],
                        outputs=[num_steps, guidance]
                    )
                    
                    with gr.Group():
                        gr.Markdown("### 🖼️ Image Settings")
                        with gr.Row():
                            image_width = gr.Slider(
                                minimum=256,
                                maximum=1024,
                                step=64,
                                value=512,
                                label="Width"
                            )
                            image_height = gr.Slider(
                                minimum=256,
                                maximum=1024,
                                step=64,
                                value=512,
                                label="Height"
                            )
                        
                            seed = gr.Number(
                                label="Seed",
                                value=None,
                                precision=0,
                                info="Leave empty for random seed"
                            )
                        
                        generate_btn = gr.Button(
                            "🎨 Generate Image",
                            variant="primary",
                            elem_classes="generate-btn"
                        )
                
                # Right Column - Output
                with gr.Column(scale=1):
                    output_image = gr.Image(
                        label="Generated Image",
                        type="pil",
                        show_download_button=True,
                        show_label=True
                    )
                    image_info = gr.Markdown(
                        visible=True,
                        value="*Click 'Generate Image' to create a new image*"
                    )
                    
                    # Add Stats Box
                    with gr.Group(visible=False) as stats_group:
                        gr.Markdown("### 🔍 Generation Stats")
                        with gr.Row():
                            with gr.Column(scale=1):
                                text_mem = gr.Markdown("**Text Encoding Memory:** N/A")
                                gen_mem = gr.Markdown("**Generation Memory:** N/A")
                                decode_mem = gr.Markdown("**Decoding Memory:** N/A")
                                total_mem = gr.Markdown("**Total Peak Memory:** N/A")
                            with gr.Column(scale=1):
                                text_time = gr.Markdown("**Text Encoding Time:** N/A")
                                gen_time = gr.Markdown("**Generation Time:** N/A")
                                decode_time = gr.Markdown("**Decoding Time:** N/A")
                                total_time = gr.Markdown("**Total Time:** N/A")
            
            # Event handlers
            async def on_generate(prompt, model_type, num_steps, guidance_scale, width, height, seed):
                try:
                    print("\n=== Generation Request Started ===")
                    print(f"Prompt: {prompt}")
                    print(f"Model: {model_type}")
                    print(f"Steps: {num_steps}")
                    print(f"Guidance: {guidance_scale}")
                    print(f"Size: {width}x{height}")
                    print(f"Seed: {seed}")

                    # Reset peak memory tracking
                    mx.metal.reset_peak_memory()

                    # Track timing
                    import time
                    start_total = time.time()

                    # Create API request
                    request = SDAPIRequest(
                        prompt=prompt,
                        model=model_type,
                        width=width,
                        height=height,
                        steps=num_steps,
                        cfg_scale=guidance_scale,
                        seed=seed if seed is not None else -1,
                        batch_size=1,
                        n_iter=1
                    )
                    
                    # Initialize pipeline if needed
                    start_text = time.time()
                    pipeline = api.init_pipeline(model_type)
                    
                    # Text encoding (includes pipeline initialization)
                    text_mem = mx.metal.get_peak_memory() / 1024**3
                    text_time = time.time() - start_text
                    mx.metal.reset_peak_memory()
                    
                    # Generation
                    start_gen = time.time()
                    latents = pipeline.generate_latents(
                        prompt,
                        n_images=1,
                        num_steps=num_steps or (50 if model_type == "dev" else 2),
                        latent_size=(height // 8, width // 8),
                        guidance=guidance_scale,
                        seed=seed if seed is not None else None
                    )
                    
                    # Process latents
                    conditioning = next(latents)
                    mx.eval(conditioning)
                    
                    for x_t in latents:
                        mx.eval(x_t)
                    
                    gen_mem = mx.metal.get_peak_memory() / 1024**3
                    gen_time = time.time() - start_gen
                    mx.metal.reset_peak_memory()
                    
                    # Decoding
                    start_decode = time.time()
                    decoded = []
                    for i in range(1):  # Single image for now
                        decoded.append(pipeline.decode(x_t[i:i+1], (height // 8, width // 8)))
                        mx.eval(decoded[-1])
                    
                    decode_mem = mx.metal.get_peak_memory() / 1024**3
                    decode_time = time.time() - start_decode
                    total_time = time.time() - start_total
                    
                    # Convert to PIL Image
                    img_array = (mx.array(decoded[0][0]) * 255).astype(mx.uint8)
                    pil_image = Image.fromarray(np.array(img_array))
                    
                    # Format stats strings
                    stats = {
                        "text_mem": f"**Text Encoding Memory:** {text_mem:.2f}GB",
                        "gen_mem": f"**Generation Memory:** {gen_mem:.2f}GB",
                        "decode_mem": f"**Decoding Memory:** {decode_mem:.2f}GB",
                        "total_mem": f"**Total Peak Memory:** {max(text_mem, gen_mem, decode_mem):.2f}GB",
                        "text_time": f"**Text Encoding Time:** {text_time:.2f}s",
                        "gen_time": f"**Generation Time:** {gen_time:.2f}s",
                        "decode_time": f"**Decoding Time:** {decode_time:.2f}s",
                        "total_time": f"**Total Time:** {total_time:.2f}s"
                    }
                    
                    print("Generation completed successfully")
                    print(f"Memory usage: Text={text_mem:.2f}GB, Gen={gen_mem:.2f}GB, Decode={decode_mem:.2f}GB")
                    print(f"Timing: Text={text_time:.2f}s, Gen={gen_time:.2f}s, Decode={decode_time:.2f}s, Total={total_time:.2f}s")
                    
                    return [
                        pil_image,  # Image
                        gr.Markdown(value=f"Generated with Flux {model_type} model"),  # Info
                        gr.Group(visible=True),  # Stats group visibility
                        *[gr.Markdown(value=v) for v in stats.values()]  # Stats values
                    ]
                except Exception as e:
                    print(f"Exception in on_generate: {str(e)}")
                    return [
                        None,  # Image
                        gr.Markdown(visible=True, value=f"❌ Error: {str(e)}"),  # Info
                        gr.Group(visible=False),  # Hide stats
                        *[gr.Markdown(value="N/A") for _ in range(8)]  # Reset stats
                    ]
                finally:
                    print("=== Generation Request Ended ===\n")
            
            # Connect event handlers
            generate_btn.click(
                fn=on_generate,
                inputs=[
                    prompt,
                    model_type,
                    num_steps,
                    guidance,
                    image_width,
                    image_height,
                    seed
                ],
                outputs=[
                    output_image,
                    image_info,
                    stats_group,
                    text_mem,
                    gen_mem,
                    decode_mem,
                    total_mem,
                    text_time,
                    gen_time,
                    decode_time,
                    total_time
                ]
            )
    
    return blocks

def main():
    """Main entry point"""
    try:
        check_system_compatibility()
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="FLUX Image Generator")
        parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
        
        # Add mutually exclusive group for listening options
        listen_group = parser.add_mutually_exclusive_group()
        listen_group.add_argument("--listen-all", action="store_true", help="Listen on all network interfaces (0.0.0.0)")

        args = parser.parse_args()
        
        # Determine host based on listening flags
        if args.listen_all:
            host = "0.0.0.0"
            print("\nWarning: Server is listening on all network interfaces (0.0.0.0)")
            print("This mode is less secure and should only be used in trusted networks")
        else:
            host = "127.0.0.1"  # localhost only
            print("\nServer is listening on localhost only (most secure)")
        
        # Check port availability
        if not check_port_available(host, args.port):
            port = find_available_port(host, args.port)
            print(f"\nWarning: Port {args.port} is in use, using port {port} instead")
        else:
            port = args.port
        
        # Create Gradio interface
        demo = create_ui()
        
        print(f"\nStarting Flux server on {host}:{port}")
        print("\nAccess modes:")
        print("1. Local only:     default                  (most secure, localhost only)")
        print("2. All networks:   --listen-all            (allows all connections)")
        
        if host != "127.0.0.1":
            print("\nTo use with Open WebUI in Docker:")
            print(f"1. Start the server with: python3.11 flux_app.py --listen-all")
            print(f"2. In Open WebUI, use URL: http://host.docker.internal:{port}")
        
        # Configure Gradio queue
        demo.queue(max_size=20)  # Allow up to 20 tasks in queue
        
        # Create FastAPI app with middleware and API endpoints
        app = FastAPI()
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Create and mount API endpoints
        create_api(app)
        
        # Mount Gradio app to FastAPI
        app = gr.mount_gradio_app(app, demo, path="/")
        
        # Keep the server running
        import uvicorn
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        server.run()
        
    except SystemError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 