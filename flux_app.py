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

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# API Endpoints
@app.post("/sdapi/v1/txt2img")
async def txt2img(request: SDAPIRequest):
    try:
        images = generate_images(
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

@app.get("/sdapi/v1/sd-models")
async def list_models():
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

@app.get("/sdapi/v1/options")
async def get_options():
    """Get current options"""
    return {
        "sd_model_checkpoint": "flux-schnell",
        "sd_backend": "Flux MLX"
    }

@app.post("/sdapi/v1/options")
async def set_options(options: dict):
    """Set options (stub for compatibility)"""
    return {"success": True}

@app.get("/sdapi/v1/progress")
async def get_progress():
    """Get generation progress (stub for compatibility)"""
    return {
        "progress": 0,
        "eta_relative": 0,
        "state": {"skipped": False, "interrupted": False, "job": "", "job_count": 0, "job_timestamp": ""},
        "current_image": None,
        "textinfo": "Idle"
    }

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

def init_pipeline(model: str):
    global flux_pipeline
    
    if flux_pipeline is None:
        model_name = "flux-" + model
        flux_pipeline = FluxPipeline(model_name)
    return flux_pipeline

def generate_images(
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
    # Use original Flux pipeline
    pipeline = init_pipeline(model)
    
    # Parse image size
    latent_size = (height // 8, width // 8)
    steps = steps or (50 if model == "dev" else 2)

    # Use original Flux pipeline
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
                                minimum=2,
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
                                num_steps: gr.update(value=50, minimum=2, maximum=100, interactive=True),
                                guidance: gr.update(value=4.0, interactive=True)
                            }
                        else:  # schnell
                            return {
                                num_steps: gr.update(value=2, minimum=2, maximum=100, interactive=True),
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
            
            # Event handlers
            def on_generate(prompt, model_type, num_steps, guidance_scale, width, height, seed):
                try:
                    print("\n=== Generation Request Started ===")
                    print(f"Prompt: {prompt}")
                    print(f"Model: {model_type}")
                    print(f"Steps: {num_steps}")
                    print(f"Guidance: {guidance_scale}")
                    print(f"Size: {width}x{height}")
                    print(f"Seed: {seed}")
                    
                    # Generate the image with return_pil=True for UI
                    images = generate_images(
                        prompt=prompt,
                        model=model_type,
                        width=width,
                        height=height,
                        steps=num_steps,
                        guidance=guidance_scale,
                        seed=seed if seed is not None else -1,
                        return_pil=True  # Return PIL Images for UI
                    )
                    
                    print("Generation completed successfully")
                    
                    if images:
                        return [
                            images[0],  # Gradio Image component can handle PIL Image directly
                            gr.Markdown(visible=True, value="✨ Generation successful!")
                        ]
                    else:
                        return [
                            None,
                            gr.Markdown(visible=True, value="❌ No images generated")
                        ]
                            
                except Exception as e:
                    print(f"Exception in on_generate: {str(e)}")
                    return [
                        None,
                        gr.Markdown(visible=True, value=f"❌ Error: {str(e)}")
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
                    image_info
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

def run_api(host: str, port: int):
    """Run the FastAPI server"""
    import uvicorn
    try:
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        print(f"Error starting API server: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    try:
        check_system_compatibility()
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="FLUX Image Generator")
        parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
        parser.add_argument("--port", type=int, default=7861, help="Port to run the UI server on")
        parser.add_argument("--api-port", type=int, default=7860, help="Port for the API server")
        
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
        
        # Check port availability and find alternative ports if needed
        try:
            api_port = find_available_port(host, args.api_port)
            ui_port = find_available_port(host, args.port)
            
            if api_port != args.api_port:
                print(f"\nWarning: API port {args.api_port} is in use, using port {api_port} instead")
            if ui_port != args.port:
                print(f"Warning: UI port {args.port} is in use, using port {ui_port} instead")
        except RuntimeError as e:
            print(f"Error: {e}")
            print("Please try different port numbers or free up some ports")
            sys.exit(1)
        
        # Create Gradio interface
        demo = create_ui()
        
        print(f"\nStarting Flux servers:")
        print(f"1. API server: {host}:{api_port}")
        print(f"2. UI server:  {host}:{ui_port}")
        print("\nAccess modes:")
        print("1. Local only:     default                  (most secure, localhost only)")
        print("2. Local network:  --listen-local          (allows LAN access)")
        print("3. All networks:   --listen-all            (allows all connections)")
        
        if host != "127.0.0.1":
            print("\nTo use with Open WebUI in Docker:")
            print(f"1. Start the server with: python3.11 flux_app.py --listen-all")
            print(f"2. In Open WebUI, use URL: http://host.docker.internal:{api_port}")
        
        # Configure Gradio
        demo.queue(max_size=20)  # Allow up to 20 tasks in queue
        
        # Start FastAPI server in a separate process
        import multiprocessing
        api_process = multiprocessing.Process(target=run_api, args=(host, api_port))
        api_process.start()
        
        # Start Gradio UI
        try:
            demo.launch(
                server_name=host,
                server_port=ui_port,
                share=False,  # Avoid antivirus issues
                show_error=True,
                inbrowser=True,
                max_threads=1  # Limit to one concurrent job
            )
        except Exception as e:
            print(f"Error starting UI server: {e}")
            api_process.terminate()
            api_process.join()
            sys.exit(1)
        finally:
            # Clean up
            api_process.terminate()
            api_process.join()
        
    except SystemError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 