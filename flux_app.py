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

def check_model_status(model_name: str) -> str:
    """Check if model files exist and return status message"""
    try:
        # Use original Flux pipeline
        pipeline = FluxPipeline(model_name)
        pipeline.ensure_models_are_loaded()
        return "✅ Downloaded"
    except Exception as e:
        print(f"Error checking model status: {str(e)}")
        return "❌ Not downloaded"

def download_model_ui(model_name: str, force: bool = False) -> str:
    """Download model files from UI"""
    try:
        # Use original Flux pipeline
        pipeline = FluxPipeline(model_name)
        pipeline.ensure_models_are_loaded()
        return f"✅ Successfully downloaded {model_name} model"
    except Exception as e:
        return f"❌ Error downloading {model_name} model: {str(e)}"

def check_and_download_models(model_name: str, force_download: bool = False):
    """Download model files from HuggingFace Hub if they don't exist"""
    config = configs[model_name]
    if config.repo_id is None:
        raise ValueError(f"No repository configured for model {model_name}")
    
    # Check in HuggingFace cache directory
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache = cache_dir / f"models--black-forest-labs--{model_name}"
    
    # Create directories if they don't exist
    model_cache.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download model files
        print(f"Downloading {model_name} model files...")
        
        # Download flow model
        if force_download or not (model_cache / "snapshots" / "main" / f"{model_name}-flow.safetensors").exists():
            flow_path = hf_hub_download(config.repo_id, config.repo_flow)
            print(f"✓ Downloaded flow model: {flow_path}")
        
        # Download autoencoder
        if force_download or not (model_cache / "snapshots" / "main" / "ae.safetensors").exists():
            ae_path = hf_hub_download(config.repo_id, config.repo_ae)
            print(f"✓ Downloaded autoencoder: {ae_path}")
        
        # Download CLIP text encoder
        if force_download or not (model_cache / "text_encoder").exists():
            clip_config = hf_hub_download(config.repo_id, "text_encoder/config.json")
            clip_model = hf_hub_download(config.repo_id, "text_encoder/model.safetensors")
            print(f"✓ Downloaded CLIP text encoder")
        
        # Download T5 text encoder
        if force_download or not (model_cache / "text_encoder_2").exists():
            t5_config = hf_hub_download(config.repo_id, "text_encoder_2/config.json")
            t5_model_index = hf_hub_download(config.repo_id, "text_encoder_2/model.safetensors.index.json")
            
            # Load model index to get weight files
            with open(t5_model_index) as f:
                weight_files = set()
                for _, w in json.load(f)["weight_map"].items():
                    weight_files.add(w)
            
            # Download each weight file
            for w in weight_files:
                w = f"text_encoder_2/{w}"
                hf_hub_download(config.repo_id, w)
            print(f"✓ Downloaded T5 text encoder")
        
        # Set environment variables for model paths
        if model_name == "flux-schnell":
            os.environ["FLUX_SCHNELL"] = str(model_cache / "snapshots" / "main" / "flux1-schnell.safetensors")
        elif model_name == "flux-dev":
            os.environ["FLUX_DEV"] = str(model_cache / "snapshots" / "main" / "flux1-dev.safetensors")
        
        # Set autoencoder path
        os.environ["AE"] = str(model_cache / "snapshots" / "main" / "ae.safetensors")
        
        print(f"✅ Successfully downloaded and configured {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {str(e)}")
        raise

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
        .model-status { margin: 10px 0; padding: 10px; border-radius: 5px; background: #f5f5f5; }
    """
    
    blocks = gr.Blocks(theme=gr.themes.Soft(), css=css)
    with blocks:
        # Model Management in Sidebar
        with gr.Sidebar():
            gr.Markdown("### 📦 Model Management")
            
            # Flux Schnell Model
            with gr.Group(elem_classes="model-status"):
                gr.Markdown("#### Flux Schnell")
                schnell_status = gr.Markdown(value=check_model_status("flux-schnell"))
                with gr.Row():
                    schnell_download = gr.Button("📥 Download Model")
                    schnell_force = gr.Checkbox(label="Force Download", value=False)
            
            # Flux Dev Model
            with gr.Group(elem_classes="model-status"):
                gr.Markdown("#### Flux Dev")
                dev_status = gr.Markdown(value=check_model_status("flux-dev"))
                with gr.Row():
                    dev_download = gr.Button("📥 Download Model")
                    dev_force = gr.Checkbox(label="Force Download", value=False)
            
            download_status = gr.Textbox(
                label="Download Status",
                value="",
                lines=2,
                interactive=False
            )
        
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
                    
                    # Call API
                    response = await api.txt2img(request)
                    
                    # Convert base64 to PIL Image
                    image_data = base64.b64decode(response.images[0].split(",")[1])
                    pil_image = Image.open(io.BytesIO(image_data))
                    
                    # Get pipeline instance and stats
                    pipeline = api.pipeline
                    
                    # Get peak memory usage from MLX
                    text_mem = mx.metal.get_peak_memory() / 1024**3
                    mx.metal.reset_peak_memory()
                    gen_mem = mx.metal.get_peak_memory() / 1024**3
                    mx.metal.reset_peak_memory()
                    decode_mem = mx.metal.get_peak_memory() / 1024**3
                    total_mem = max(text_mem, gen_mem, decode_mem)
                    
                    # Format stats strings
                    stats = {
                        "text_mem": f"**Text Encoding Memory:** {text_mem:.2f}GB",
                        "gen_mem": f"**Generation Memory:** {gen_mem:.2f}GB",
                        "decode_mem": f"**Decoding Memory:** {decode_mem:.2f}GB",
                        "total_mem": f"**Total Peak Memory:** {total_mem:.2f}GB",
                        "text_time": f"**Text Encoding Time:** {0.0:.2f}s",
                        "gen_time": f"**Generation Time:** {0.0:.2f}s",
                        "decode_time": f"**Decoding Time:** {0.0:.2f}s",
                        "total_time": f"**Total Time:** {0.0:.2f}s"
                    }
                    
                    print("Generation completed successfully")
                    
                    return [
                        pil_image,  # Image
                        gr.Markdown(value=response.info),  # Info
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
            
            def on_model_download(model_name: str, force: bool = False) -> Tuple[str, str]:
                """Handle model download and update status"""
                try:
                    check_and_download_models(model_name, force)
                    status = check_model_status(model_name)
                    return [
                        status,
                        f"✅ Successfully downloaded {model_name} model"
                    ]
                except Exception as e:
                    return [
                        check_model_status(model_name),
                        f"❌ Error downloading {model_name} model: {str(e)}"
                    ]
            
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
            
            # Model download handlers
            schnell_download.click(
                fn=lambda force: on_model_download("flux-schnell", force),
                inputs=[schnell_force],
                outputs=[schnell_status, download_status]
            )
            
            dev_download.click(
                fn=lambda force: on_model_download("flux-dev", force),
                inputs=[dev_force],
                outputs=[dev_status, download_status]
            )
    
    return blocks

def main():
    """Main entry point"""
    try:
        check_system_compatibility()
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="FLUX Image Generator")
        parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
        parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
        
        # Add mutually exclusive group for listening options
        listen_group = parser.add_mutually_exclusive_group()
        listen_group.add_argument("--listen-all", action="store_true", help="Listen on all network interfaces (0.0.0.0)")
        listen_group.add_argument("--listen-local", action="store_true", help="Listen on local network (192.168.0.0/16, 10.0.0.0/8)")
        
        # Add model download options
        parser.add_argument("--download-models", action="store_true", help="Download models if not present")
        parser.add_argument("--force-download", action="store_true", help="Force re-download of models even if present")
        
        args = parser.parse_args()
        
        # Check and download models if requested
        if args.download_models or args.force_download:
            print("\nChecking model files...")
            for model in ["flux-schnell", "flux-dev"]:
                check_and_download_models(model, args.force_download)
        
        # Determine host based on listening flags
        if args.listen_all:
            host = "0.0.0.0"
            print("\nWarning: Server is listening on all network interfaces (0.0.0.0)")
            print("This mode is less secure and should only be used in trusted networks")
        elif args.listen_local:
            host = "192.168.1.1"  # This will allow local network access
            print("\nWarning: Server is listening on local network")
            print("This mode allows access from devices on your local network")
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
        print("2. Local network:  --listen-local          (allows LAN access)")
        print("3. All networks:   --listen-all            (allows all connections)")
        
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