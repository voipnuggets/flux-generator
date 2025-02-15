import io
import base64
import argparse
import gradio as gr
from typing import Optional, List, Tuple
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

app = FastAPI(title="Flux Image Generator")

# Add CORS middleware for API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
flux_pipeline = None

class GenerationRequest(BaseModel):
    prompt: str
    model: str = "schnell"
    n_images: int = 1
    image_size: str = "512x512"
    steps: Optional[int] = None
    guidance: float = 4.0
    seed: Optional[int] = None
    adapter_path: Optional[str] = None
    fuse_adapter: bool = False
    quantize: bool = False

class GenerationResponse(BaseModel):
    images: List[str]
    peak_memory: Optional[float] = None

class SDAPIRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    seed: int = -1
    cfg_scale: float = 4.0
    steps: int = 2
    batch_size: int = 1
    n_iter: int = 1
    model: str = "schnell"

class SDAPIResponse(BaseModel):
    images: List[str]
    parameters: dict
    info: str

class SDModelResponse(BaseModel):
    title: str
    model_name: str
    hash: str = ""
    sha256: str = ""
    filename: str = ""
    config: Optional[dict] = None

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

def generate_images(request: GenerationRequest) -> List[str]:
    init_pipeline(request.model, request.quantize)
    
    # Parse image size
    width, height = map(int, request.image_size.split('x'))
    latent_size = to_latent_size((height, width))
    
    if request.adapter_path:
        load_adapter(flux_pipeline, request.adapter_path, fuse=request.fuse_adapter)

    steps = request.steps or (50 if request.model == "dev" else 2)
    
    # Generate images
    latents = flux_pipeline.generate_latents(
        request.prompt,
        n_images=request.n_images,
        num_steps=steps,
        latent_size=latent_size,
        guidance=request.guidance,
        seed=request.seed
    )

    # Process latents
    conditioning = next(latents)
    mx.eval(conditioning)
    
    for x_t in latents:
        mx.eval(x_t)

    # Decode images
    decoded = []
    for i in range(0, request.n_images):
        decoded.append(flux_pipeline.decode(x_t[i:i+1], latent_size))
        mx.eval(decoded[-1])

    # Convert to base64
    images = []
    for img_tensor in decoded:
        img_array = (mx.array(img_tensor[0]) * 255).astype(mx.uint8)
        pil_image = Image.fromarray(np.array(img_array))
        
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        images.append(f"data:image/png;base64,{img_str}")
    
    return images

@app.post("/generate")
async def generate(request: GenerationRequest) -> GenerationResponse:
    images = generate_images(request)
    return GenerationResponse(images=images)

@app.post("/generate_form")
async def generate_form(
    prompt: str = Form(...),
    model: str = Form(default="schnell"),
    n_images: str = Form(default="1"),
    image_size: str = Form(default="512x512"),
    steps: str = Form(default=""),
    guidance: str = Form(default="4.0"),
    seed: str = Form(default=""),
    adapter_path: str = Form(default=""),
    fuse_adapter: str = Form(default="false"),
    quantize: str = Form(default="false")
):
    # Convert form values to appropriate types
    try:
        n_images_int = int(n_images)
        steps_int = int(steps) if steps.strip() else None
        guidance_float = float(guidance)
        seed_int = int(seed) if seed.strip() else None
        fuse_adapter_bool = fuse_adapter.lower() == "true"
        quantize_bool = quantize.lower() == "true"
        
        request = GenerationRequest(
            prompt=prompt,
            model=model,
            n_images=n_images_int,
            image_size=image_size,
            steps=steps_int,
            guidance=guidance_float,
            seed=seed_int,
            adapter_path=adapter_path if adapter_path.strip() else None,
            fuse_adapter=fuse_adapter_bool,
            quantize=quantize_bool
        )
        images = generate_images(request)
        return {"images": images}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid form data: {str(e)}")

@app.post("/sdapi/v1/txt2img")
async def txt2img(request: SDAPIRequest) -> SDAPIResponse:
    """
    Stable Diffusion Web UI compatible txt2img endpoint
    """
    # Convert SD API request to our internal format
    gen_request = GenerationRequest(
        prompt=request.prompt,
        model=request.model,
        n_images=request.batch_size * request.n_iter,
        image_size=f"{request.width}x{request.height}",
        steps=request.steps,
        guidance=request.cfg_scale,
        seed=None if request.seed == -1 else request.seed,
    )
    
    # Generate images
    images = generate_images(gen_request)
    
    # Format response in SD API format
    parameters = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "width": request.width,
        "height": request.height,
        "seed": request.seed,
        "cfg_scale": request.cfg_scale,
        "steps": request.steps,
        "batch_size": request.batch_size,
        "n_iter": request.n_iter,
    }
    
    info = f"Generated with Flux {request.model} model"
    
    return SDAPIResponse(
        images=images,
        parameters=parameters,
        info=info
    )

def generate_for_ui(prompt, model_type, num_steps, guidance_scale, width, height, seed):
    """Wrapper function for generate_images to work with Gradio UI"""
    request = GenerationRequest(
        prompt=prompt,
        model=model_type,
        n_images=1,
        image_size=f"{width}x{height}",
        steps=num_steps if num_steps else None,
        guidance=guidance_scale,
        seed=seed if seed else None
    )
    images = generate_images(request)
    return images[0] if images else None

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

def create_ui(enable_api: bool = False, api_port: int = 7860):
    """Create the Gradio interface"""
    css = """
        .container { max-width: 1200px; margin: auto; }
        .prompt-box { min-height: 100px; }
        .status-box { margin-top: 10px; }
        .generate-btn { min-height: 60px; }
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
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
                
                # API Documentation Tab
                if enable_api:
                    with gr.Tab("🔧 API", id="api_tab"):
                        gr.Markdown(
                            """
                            ### API Documentation
                            The Flux Image Generator provides a Stable Diffusion-compatible API
                            that can be used with third-party UIs like Open WebUI.
                            
                            #### Endpoints:
                            - `/sdapi/v1/txt2img` - Generate images
                            - `/sdapi/v1/sd-models` - List available models
                            - `/sdapi/v1/options` - Get/set generation options
                            """
                        )
                        api_btn = gr.Button(
                            "📚 View Full API Documentation",
                            variant="secondary"
                        )
                        api_btn.click(
                            lambda: webbrowser.open(f"http://127.0.0.1:{api_port}/docs"),
                            None,
                            None
                        )
            
            # Event handlers
            def on_generate(*args):
                try:
                    image = generate_for_ui(*args)
                    if image:
                        return [
                            image,
                            gr.Markdown(visible=True, value="✨ Image generated successfully!")
                        ]
                    else:
                        return [
                            None,
                            gr.Markdown(visible=True, value="❌ Failed to generate image")
                        ]
                except Exception as e:
                    return [
                        None,
                        gr.Markdown(visible=True, value=f"❌ Error: {str(e)}")
                    ]
            
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
    
    return demo

@app.get("/sdapi/v1/sd-models")
async def get_models():
    """Get available models endpoint required by Open WebUI"""
    print("Models endpoint called")  # Debug log
    models = [
        {
            "title": "Flux Schnell",
            "model_name": "flux-schnell",
            "hash": "",
            "sha256": "",
            "filename": "flux-schnell",
            "config": None
        },
        {
            "title": "Flux Dev",
            "model_name": "flux-dev",
            "hash": "",
            "sha256": "",
            "filename": "flux-dev",
            "config": None
        }
    ]
    return models

@app.get("/sdapi/v1/options")
async def get_options():
    """Get options endpoint required by Open WebUI"""
    return {
        "sd_model_checkpoint": "flux-schnell",
        "sd_vae": "None",
        "CLIP_stop_at_last_layers": 1,
        "img2img_fix_steps": False,
        "enable_emphasis": True,
        "enable_batch_seeds": True,
        "token_merging_ratio": 0.0,
        "sd_backend": "Flux MLX"
    }

@app.post("/sdapi/v1/options")
async def set_options(options: dict):
    """Set options endpoint required by Open WebUI"""
    return {"message": "Options updated"}

@app.get("/sdapi/v1/cmd-flags")
async def get_cmd_flags():
    """Get command line flags endpoint required by Open WebUI"""
    return {
        "api": True,
        "ckpt": "flux-schnell",
        "skip-torch-cuda-test": True,
        "no-half": True,
        "no-half-vae": True,
        "api-server-stop": False
    }

@app.post("/sdapi/v1/reload-checkpoint")
async def reload_checkpoint():
    """Reload model checkpoint endpoint required by Open WebUI"""
    return {"message": "Model reloaded"}

@app.get("/sdapi/v1/progress")
async def get_progress():
    """Get generation progress endpoint required by Open WebUI"""
    return {
        "progress": 0,
        "eta_relative": 0,
        "state": {
            "skipped": False,
            "interrupted": False,
            "job": "",
            "job_count": 0,
            "job_timestamp": "20240101000000",
            "job_no": 0,
            "sampling_step": 0,
            "sampling_steps": 0
        },
        "current_image": None,
        "textinfo": "Idle"
    }

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

def main():
    """Main entry point"""
    try:
        check_system_compatibility()
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="FLUX Image Generator")
        parser.add_argument("--enable-api", action="store_true", help="Enable API endpoint")
        parser.add_argument("--api-port", type=int, default=7860, help="Port for API endpoint")
        
        # Add mutually exclusive group for listening options
        listen_group = parser.add_mutually_exclusive_group()
        listen_group.add_argument("--listen-all", action="store_true", help="Listen on all network interfaces (0.0.0.0)")
        listen_group.add_argument("--listen-local", action="store_true", help="Listen on local network (192.168.0.0/16, 10.0.0.0/8)")
        listen_group.add_argument("--listen", action="store_true", help="[Deprecated] Use --listen-all instead")
        
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
        if args.listen or args.listen_all:
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
        
        if args.enable_api:
            # Use uvicorn to run FastAPI
            import uvicorn
            print(f"\nStarting Flux API server on {host}:{args.api_port}")
            print("\nAccess modes:")
            print("1. Local only:     --enable-api                  (most secure, localhost only)")
            print("2. Local network:  --enable-api --listen-local   (allows LAN access)")
            print("3. All networks:   --enable-api --listen-all     (allows all connections)")
            
            if args.listen:
                print("\nNote: --listen is deprecated, please use --listen-all instead")
            
            if host != "127.0.0.1":
                print("\nTo use with Open WebUI in Docker:")
                print("1. Start the server with: python3.11 flux_app.py --enable-api --listen-all")
                print("2. In Open WebUI, use URL: http://host.docker.internal:7860")
            
            uvicorn.run(
                app,
                host=host,
                port=args.api_port,
                reload=False,
                log_level="info"
            )
        else:
            # Create and launch UI only
            demo = create_ui(enable_api=args.enable_api, api_port=args.api_port)
            demo.launch(
                server_name=host,
                server_port=args.api_port,
                share=False,  # Avoid antivirus issues
                show_error=True,
                inbrowser=True
            )
        
    except SystemError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 