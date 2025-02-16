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

# Global pipeline instance
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

def init_pipeline(model="schnell", quantize=False):
    global flux_pipeline
    if flux_pipeline is None:
        flux_pipeline = FluxPipeline("flux-" + model)
        if quantize:
            nn.quantize(flux_pipeline.flow, class_predicate=quantization_predicate)
            nn.quantize(flux_pipeline.t5, class_predicate=quantization_predicate)
            nn.quantize(flux_pipeline.clip, class_predicate=quantization_predicate)

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
    init_pipeline(model)
    
    # Parse image size
    latent_size = to_latent_size((height, width))
    steps = steps or (50 if model == "dev" else 2)
    
    # Generate images
    latents = flux_pipeline.generate_latents(
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
        decoded.append(flux_pipeline.decode(x_t[i:i+1], latent_size))
        mx.eval(decoded[-1])

    # Convert to PIL Images or base64
    images = []
    for img_tensor in decoded:
        img_array = (mx.array(img_tensor[0]) * 255).astype(mx.uint8)
        pil_image = Image.fromarray(np.array(img_array))
        
        if return_pil:
            images.append(pil_image)
        else:
            # Convert to base64 for API
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            images.append(f"data:image/png;base64,{img_str}")
    
    return images

def check_model_status(model_name: str) -> str:
    """Check if model files exist and return status message"""
    config = configs[model_name]
    if config.repo_id is None:
        return "❌ No repository configured"
    
    models_dir = Path.home() / ".flux" / "models"
    flow_path = models_dir / f"{model_name}-flow.safetensors"
    ae_path = models_dir / f"{model_name}-ae.safetensors"
    clip_config = models_dir / model_name / "text_encoder/config.json"
    clip_model = models_dir / model_name / "text_encoder/model.safetensors"
    t5_config = models_dir / model_name / "text_encoder_2/config.json"
    t5_model_index = models_dir / model_name / "text_encoder_2/model.safetensors.index.json"
    
    files_exist = all([
        flow_path.exists(),
        ae_path.exists(),
        clip_config.exists(),
        clip_model.exists(),
        t5_config.exists(),
        t5_model_index.exists()
    ])
    
    return "✅ Downloaded" if files_exist else "❌ Not downloaded"

def download_model_ui(model_name: str, force: bool = False) -> str:
    """Download model files from UI"""
    try:
        check_and_download_models(model_name, force)
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
    """
    
    blocks = gr.Blocks(theme=gr.themes.Soft(), css=css)
    with blocks:
        with gr.Column(elem_classes="container"):
            gr.Markdown(
                """
                # 🎨 FLUX Image Generator
                Generate beautiful images from text using the FLUX model on Apple Silicon Macs.
                """
            )
            
            with gr.Tabs():
                # Model Management Tab
                with gr.Tab("📦 Model Management", id="model_tab"):
                    with gr.Column():
                        gr.Markdown("### Model Status")
                        with gr.Row():
                            with gr.Column():
                                schnell_status = gr.Textbox(
                                    label="Flux Schnell Status",
                                    value=check_model_status("flux-schnell"),
                                    interactive=False
                                )
                                download_schnell = gr.Button(
                                    "📥 Download Schnell Model",
                                    variant="primary"
                                )
                            
                            with gr.Column():
                                dev_status = gr.Textbox(
                                    label="Flux Dev Status",
                                    value=check_model_status("flux-dev"),
                                    interactive=False
                                )
                                download_dev = gr.Button(
                                    "📥 Download Dev Model",
                                    variant="primary"
                                )
                        
                        with gr.Row():
                            force_download = gr.Checkbox(
                                label="Force Re-download",
                                value=False,
                                info="Check this to re-download even if model files exist"
                            )
                        
                        download_status = gr.Textbox(
                            label="Download Status",
                            value="",
                            interactive=False,
                            elem_classes="status-box"
                        )
                
                # Image Generation Tab
                with gr.Tab("🖼️ Generate", id="generate_tab"):
                    with gr.Row():
                        # Left Column - Controls
                        with gr.Column():
                            prompt = gr.Textbox(
                                label="Prompt",
                                placeholder="Enter your prompt here... (e.g., 'a beautiful moonset over the ocean, highly detailed, 4k')",
                                lines=3,
                                elem_classes="prompt-box"
                            )
                            
                            with gr.Group():
                                gr.Markdown("### Model Settings")
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
                            
                            with gr.Group():
                                gr.Markdown("### Image Settings")
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
                        with gr.Column():
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
            download_schnell.click(
                fn=lambda force: download_model_ui("flux-schnell", force),
                inputs=[force_download],
                outputs=[download_status]
            ).then(
                fn=lambda: check_model_status("flux-schnell"),
                outputs=[schnell_status]
            )
            
            download_dev.click(
                fn=lambda force: download_model_ui("flux-dev", force),
                inputs=[force_download],
                outputs=[download_status]
            ).then(
                fn=lambda: check_model_status("flux-dev"),
                outputs=[dev_status]
            )
    
    return blocks

def check_and_download_models(model_name: str, force_download: bool = False):
    """Check if model files exist and download if needed"""
    config = configs[model_name]
    if config.repo_id is None:
        raise ValueError(f"No repository configured for model {model_name}")
    
    # Create models directory if it doesn't exist
    models_dir = Path.home() / ".flux" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Define paths for model files
    flow_path = models_dir / f"{model_name}-flow.safetensors"
    ae_path = models_dir / f"{model_name}-ae.safetensors"
    clip_config = models_dir / model_name / "text_encoder/config.json"
    clip_model = models_dir / model_name / "text_encoder/model.safetensors"
    t5_config = models_dir / model_name / "text_encoder_2/config.json"
    t5_model_index = models_dir / model_name / "text_encoder_2/model.safetensors.index.json"
    
    # Check if files exist
    files_exist = all([
        flow_path.exists(),
        ae_path.exists(),
        clip_config.exists(),
        clip_model.exists(),
        t5_config.exists(),
        t5_model_index.exists()
    ])
    
    if not files_exist or force_download:
        print(f"\nDownloading {model_name} model files...")
        
        # Download flow model
        if config.repo_flow:
            print(f"Downloading flow model...")
            flow_file = hf_hub_download(config.repo_id, config.repo_flow)
            os.rename(flow_file, flow_path)
        
        # Download autoencoder
        if config.repo_ae:
            print(f"Downloading autoencoder...")
            ae_file = hf_hub_download(config.repo_id, config.repo_ae)
            os.rename(ae_file, ae_path)
        
        # Download CLIP files
        print(f"Downloading CLIP model...")
        clip_config_file = hf_hub_download(config.repo_id, "text_encoder/config.json")
        clip_model_file = hf_hub_download(config.repo_id, "text_encoder/model.safetensors")
        
        clip_dir = models_dir / model_name / "text_encoder"
        clip_dir.mkdir(parents=True, exist_ok=True)
        os.rename(clip_config_file, clip_config)
        os.rename(clip_model_file, clip_model)
        
        # Download T5 files
        print(f"Downloading T5 model...")
        t5_config_file = hf_hub_download(config.repo_id, "text_encoder_2/config.json")
        t5_model_index_file = hf_hub_download(config.repo_id, "text_encoder_2/model.safetensors.index.json")
        
        t5_dir = models_dir / model_name / "text_encoder_2"
        t5_dir.mkdir(parents=True, exist_ok=True)
        os.rename(t5_config_file, t5_config)
        os.rename(t5_model_index_file, t5_model_index)
        
        # Download T5 model weights
        import json
        with open(t5_model_index) as f:
            weight_files = set(json.load(f)["weight_map"].values())
        
        for w in weight_files:
            print(f"Downloading T5 weight file: {w}")
            weight_file = hf_hub_download(config.repo_id, f"text_encoder_2/{w}")
            os.rename(weight_file, t5_dir / w)
        
        print(f"Model files downloaded successfully!")
    
    # Set environment variables for model paths
    if model_name == "flux-schnell":
        os.environ["FLUX_SCHNELL"] = str(flow_path)
    elif model_name == "flux-dev":
        os.environ["FLUX_DEV"] = str(flow_path)
    os.environ["AE"] = str(ae_path)

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
        parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
        parser.add_argument("--api-port", type=int, default=7861, help="Port for the API server")
        
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
            ui_port = find_available_port(host, args.port)
            api_port = find_available_port(host, args.api_port)
            
            if ui_port != args.port:
                print(f"\nWarning: UI port {args.port} is in use, using port {ui_port} instead")
            if api_port != args.api_port:
                print(f"Warning: API port {args.api_port} is in use, using port {api_port} instead")
        except RuntimeError as e:
            print(f"Error: {e}")
            print("Please try different port numbers or free up some ports")
            sys.exit(1)
        
        # Create Gradio interface
        demo = create_ui()
        
        print(f"\nStarting Flux servers:")
        print(f"1. UI server:  {host}:{ui_port}")
        print(f"2. API server: {host}:{api_port}")
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