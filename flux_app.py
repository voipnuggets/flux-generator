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

# Add directories to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "stable_diffusion"))
sys.path.append(os.path.join(SCRIPT_DIR, "musicgen"))

from stable_diffusion import StableDiffusion, StableDiffusionXL
from musicgen.musicgen import MusicGen
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json

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
    model: str = "schnell"  # Options: "schnell", "dev", "flux-schnell", "flux-dev", "stabilityai/stable-diffusion-2-1-base", "stabilityai/sdxl-turbo"

class SDAPIResponse(BaseModel):
    images: List[str]
    parameters: dict
    info: str

class FluxAPI:
    """Unified API for both UI and external API calls"""
    def __init__(self):
        self.pipeline = None
        self.sd_pipeline = None
        self.current_model = None
    
    def init_pipeline(self, model: str):
        """Initialize the pipeline if needed"""
        if model.startswith("stabilityai/"):
            # Handle Stability AI models directly
            if self.sd_pipeline is None or self.current_model != model:
                if "sdxl-turbo" in model:
                    self.sd_pipeline = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
                else:
                    self.sd_pipeline = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
                self.current_model = model
            return self.sd_pipeline
        else:
            # Handle Flux models by adding 'flux-' prefix if needed
            flux_model = model if model.startswith("flux-") else f"flux-{model}"
            if self.pipeline is None or self.current_model != flux_model:
                self.pipeline = FluxPipeline(flux_model)
                self.current_model = flux_model
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
        
        if model.startswith("stabilityai/"):
            # Use stable diffusion pipeline
            steps = steps or (2 if "sdxl-turbo" in model else 50)
            guidance = guidance or (0.0 if "sdxl-turbo" in model else 7.5)
            
            # Generate latents
            latents = pipeline.generate_latents(
                prompt,
                n_images=batch_size * n_iter,
                cfg_weight=guidance,
                num_steps=steps,
                seed=seed
            )
        else:
            # Use Flux pipeline
            steps = steps or (50 if model == "flux-dev" else 2)
            
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
        if not model.startswith("stabilityai/"):
            conditioning = next(latents)
            mx.eval(conditioning)
        
        for x_t in latents:
            mx.eval(x_t)
        
        # Decode images
        decoded = []
        for i in range(0, batch_size * n_iter):
            if model.startswith("stabilityai/"):
                # Stable Diffusion decode doesn't need latent_size
                decoded.append(pipeline.decode(x_t[i:i+1]))
            else:
                # Flux decode needs latent_size
                decoded.append(pipeline.decode(x_t[i:i+1], latent_size))
            mx.eval(decoded[-1])
        
        # Convert to PIL Images or base64
        images = []
        for i, img_tensor in enumerate(decoded):
            img_array = (mx.array(img_tensor[0]) * 255).astype(mx.uint8)
            pil_image = Image.fromarray(np.array(img_array))
            
            if return_pil:
                images.append(pil_image)
            else:
                # Convert to base64 for API response
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                images.append(img_str)
        
        return images
    
    def list_models(self):
        """List available models"""
        return [
            {
                "title": "flux-schnell",
                "name": "Flux Schnell (Fast)",
                "model_name": "flux-schnell",
                "hash": None,
                "sha256": None,
                "filename": "flux-schnell.safetensors",
                "config": None
            },
            {
                "title": "flux-dev",
                "name": "Flux Dev (High Quality)",
                "model_name": "flux-dev",
                "hash": None,
                "sha256": None,
                "filename": "flux-dev.safetensors",
                "config": None
            },
            {
                "title": "stabilityai/stable-diffusion-2-1-base",
                "name": "SD 2.1 Base (High Quality)",
                "model_name": "stabilityai/stable-diffusion-2-1-base",
                "hash": None,
                "sha256": None,
                "filename": "sd-2-1-base.safetensors",
                "config": None
            },
            {
                "title": "stabilityai/sdxl-turbo",
                "name": "SDXL Turbo (Fast)",
                "model_name": "stabilityai/sdxl-turbo",
                "hash": None,
                "sha256": None,
                "filename": "sdxl-turbo.safetensors",
                "config": None
            }
        ]
    
    def get_options(self):
        """Get current options"""
        return {
            "sd_model_checkpoint": "stabilityai/stable-diffusion-2-1-base",  # Set SD 2.1 as default
            "sd_backend": "Flux MLX",
            "sd_model_list": [
                {
                    "title": "Flux Schnell (Fast)",
                    "name": "flux-schnell",
                    "model_name": "flux-schnell"
                },
                {
                    "title": "SD 2.1 Base (High Quality)",
                    "name": "stabilityai/stable-diffusion-2-1-base",
                    "model_name": "stabilityai/stable-diffusion-2-1-base"
                },
                {
                    "title": "Flux Dev (High Quality)",
                    "name": "flux-dev",
                    "model_name": "flux-dev"
                },
                {
                    "title": "SDXL Turbo (Fast)",
                    "name": "stabilityai/sdxl-turbo",
                    "model_name": "stabilityai/sdxl-turbo"
                }
            ]
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

def create_musicgen_ui():
    """Create the MusicGen UI interface"""
    with gr.Column():
        gr.Markdown(
            """
            # 🎵 MusicGen
            Generate music using Apple Silicon optimized models.
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                text_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your music description here...",
                    lines=3,
                    interactive=True
                )
                
                # Generation Parameters
                with gr.Group():
                    gr.Markdown("#### ⚙️ Generation Parameters")
                    with gr.Row():
                        with gr.Column(scale=1):
                            max_steps = gr.Slider(
                                minimum=50,
                                maximum=500,
                                step=50,
                                value=200,
                                label="Max Steps"
                            )
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                step=0.1,
                                value=1.0,
                                label="Temperature"
                            )
                        with gr.Column(scale=1):
                            top_k = gr.Slider(
                                minimum=50,
                                maximum=500,
                                step=50,
                                value=250,
                                label="Top K"
                            )
                            guidance = gr.Slider(
                                minimum=1.0,
                                maximum=10.0,
                                step=0.5,
                                value=3.0,
                                label="Guidance Scale"
                            )
                
                generate_btn = gr.Button("Generate Music", variant="primary")

            with gr.Column(scale=6):
                audio_output = gr.Audio(label="Generated Audio", interactive=False)
                generation_info = gr.Markdown(visible=True, value="*Click 'Generate' to create music*")

                # Add Stats Box
                with gr.Group(visible=True) as stats_group:
                    gr.Markdown("### 🔍 Generation Stats")
                    with gr.Row():
                        with gr.Column(scale=1):
                            gen_mem = gr.Markdown("**Generation Memory:** N/A")
                            total_mem = gr.Markdown("**Total Peak Memory:** N/A")
                        with gr.Column(scale=1):
                            gen_time = gr.Markdown("**Generation Time:** N/A")
                            total_time = gr.Markdown("**Total Time:** N/A")

        def generate_music(prompt, max_steps, top_k, temperature, guidance):
            try:
                # Reset peak memory tracking
                mx.metal.reset_peak_memory()
                
                # Track timing
                import time
                start_total = time.time()
                
                # Initialize model
                model = MusicGen.from_pretrained("facebook/musicgen-medium")
                
                # Generate audio
                start_gen = time.time()
                audio = model.generate(
                    text=prompt,
                    max_steps=max_steps,
                    top_k=top_k,
                    temp=temperature,
                    guidance_coef=guidance
                )
                
                gen_mem = mx.metal.get_peak_memory() / 1024**3
                gen_time = time.time() - start_gen
                total_time = time.time() - start_total
                
                # Convert audio to WAV format
                import scipy.io.wavfile as wav
                import tempfile
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    wav.write(temp_file.name, model.sampling_rate, audio)
                    audio_path = temp_file.name
                
                return [
                    (audio_path, model.sampling_rate),  # Audio output
                    f"Generated music from prompt: {prompt}",  # Info
                    f"**Generation Memory:** {gen_mem:.2f}GB",  # gen_mem
                    f"**Total Peak Memory:** {gen_mem:.2f}GB",  # total_mem
                    f"**Generation Time:** {gen_time:.2f}s",  # gen_time
                    f"**Total Time:** {total_time:.2f}s"  # total_time
                ]
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                return [
                    None,  # Audio output
                    error_msg,  # Info
                    "**Generation Memory:** N/A",  # gen_mem
                    "**Total Peak Memory:** N/A",  # total_mem
                    "**Generation Time:** N/A",  # gen_time
                    "**Total Time:** N/A"  # total_time
                ]

        # Wire up the generate button
        generate_btn.click(
            fn=generate_music,
            inputs=[
                text_prompt,
                max_steps,
                top_k,
                temperature,
                guidance
            ],
            outputs=[
                audio_output,
                generation_info,
                gen_mem,
                total_mem,
                gen_time,
                total_time
            ]
        )

    return gr.Column()

def create_ui():
    """Create the Gradio UI interface"""
    with gr.Blocks(title="Flux Generator") as demo:
        with gr.Tabs():
            with gr.Tab("🖼️ Image Generation"):
                gr.Markdown(
                    """
                    # 🌊 Flux Generator
                    Generate beautiful images using Apple Silicon optimized models.
                    """
                )
                
                with gr.Row():
                    with gr.Column(scale=4):
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here...",
                            lines=3,
                            interactive=True
                        )
                        
                        # Model Selection - Changed from Radio to Dropdown
                        gr.Markdown("### 🎨 Model Selection")
                        model_choices = [
                            "Flux Schnell (Fast, 2 steps)",
                            "Flux Dev (High Quality, 50 steps)",
                            "SD 2.1 Base (High Quality)",
                            "SDXL Turbo (Fast)"
                        ]
                        model = gr.Dropdown(
                            choices=model_choices,
                            value="Flux Schnell (Fast, 2 steps)",
                            label="Select Model",
                            interactive=True
                        )
                        
                        # Generation Parameters
                        with gr.Group():
                            gr.Markdown("#### ⚙️ Generation Parameters")
                            with gr.Row():
                                with gr.Column(scale=1):
                                    num_steps = gr.Slider(
                                        minimum=1,
                                        maximum=100,
                                        step=1,
                                        value=2,
                                        label="Steps"
                                    )
                                    guidance_scale = gr.Slider(
                                        minimum=0.0,
                                        maximum=20.0,
                                        step=0.5,
                                        value=4.0,
                                        label="Guidance Scale"
                                    )
                                with gr.Column(scale=1):
                                    with gr.Row():
                                        width = gr.Slider(
                                            minimum=256,
                                            maximum=1024,
                                            step=64,
                                            value=512,
                                            label="Width"
                                        )
                                        height = gr.Slider(
                                            minimum=256,
                                            maximum=1024,
                                            step=64,
                                            value=512,
                                            label="Height"
                                        )
                                    seed = gr.Number(
                                        value=-1,
                                        label="Seed (-1 for random)",
                                        precision=0
                                    )
                        
                        generate = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=6):
                        result = gr.Image(label="Generated Image", interactive=False)
                        image_info = gr.Markdown(visible=True, value="*Click 'Generate' to create a new image*")

                        # Add Stats Box
                        with gr.Group(visible=True) as stats_group:
                            gr.Markdown("### 🔍 Generation Stats")
                            with gr.Row():
                                with gr.Column(scale=1):
                                    text_mem = gr.Markdown("**Text Encoding Memory:** N/A")
                                    gen_mem = gr.Markdown("**Generation Memory:** N/A")
                                    total_mem = gr.Markdown("**Total Peak Memory:** N/A")
                                with gr.Column(scale=1):
                                    text_time = gr.Markdown("**Text Encoding Time:** N/A")
                                    gen_time = gr.Markdown("**Generation Time:** N/A")
                                    total_time = gr.Markdown("**Total Time:** N/A")

            with gr.Tab("🎵 Music Generation"):
                create_musicgen_ui()

        def update_model_params(model_choice):
            """Update parameters based on selected model"""
            if "Schnell" in model_choice:
                return 2, 4.0  # steps, guidance
            elif "Dev" in model_choice:
                return 50, 4.0
            elif "2.1" in model_choice:
                return 50, 7.5
            else:
                return 2, 0.0

        def generate_with_stats(prompt, model_choice, steps, guidance, width, height, seed):
            try:
                # Get the actual model key based on selection
                if "Schnell" in model_choice:
                    model = "flux-schnell"
                elif "Dev" in model_choice:
                    model = "flux-dev"
                elif "2.1" in model_choice:
                    model = "stabilityai/stable-diffusion-2-1-base"
                else:
                    model = "stabilityai/sdxl-turbo"
                
                # Reset peak memory tracking
                mx.metal.reset_peak_memory()

                # Track timing
                import time
                start_total = time.time()

                # Initialize pipeline
                start_text = time.time()
                pipeline = api.init_pipeline(model)
                text_mem = mx.metal.get_peak_memory() / 1024**3
                text_time = time.time() - start_text
                mx.metal.reset_peak_memory()

                # Generate image
                start_gen = time.time()
                image_b64 = api.generate_images(
                    prompt=prompt,
                    model=model,
                    steps=steps,
                    guidance=guidance,
                    width=width,
                    height=height,
                    seed=seed if seed >= 0 else None
                )[0]

                # Convert base64 back to PIL Image for Gradio
                if image_b64 and not isinstance(image_b64, Image.Image):
                    image_bytes = base64.b64decode(image_b64)
                    image = Image.open(io.BytesIO(image_bytes))
                else:
                    image = image_b64

                gen_mem = mx.metal.get_peak_memory() / 1024**3
                gen_time = time.time() - start_gen
                total_time = time.time() - start_total

                # Get friendly model name
                model_name = model_choice

                return [
                    image,  # Image
                    f"Generated with {model_name}",  # Info
                    f"**Text Encoding Memory:** {text_mem:.2f}GB",  # text_mem
                    f"**Generation Memory:** {gen_mem:.2f}GB",  # gen_mem
                    f"**Total Peak Memory:** {max(text_mem, gen_mem):.2f}GB",  # total_mem
                    f"**Text Encoding Time:** {text_time:.2f}s",  # text_time
                    f"**Generation Time:** {gen_time:.2f}s",  # gen_time
                    f"**Total Time:** {total_time:.2f}s"  # total_time
                ]
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                return [
                    None,  # Image
                    error_msg,  # Info
                    "**Text Encoding Memory:** N/A",  # text_mem
                    "**Generation Memory:** N/A",  # gen_mem
                    "**Total Peak Memory:** N/A",  # total_mem
                    "**Text Encoding Time:** N/A",  # text_time
                    "**Generation Time:** N/A",  # gen_time
                    "**Total Time:** N/A"  # total_time
                ]

        # Update the model parameter changes
        model.change(
            fn=update_model_params,
            inputs=[model],
            outputs=[num_steps, guidance_scale]
        )

        # Update the generate click handler
        generate.click(
            fn=generate_with_stats,
            inputs=[
                prompt,
                model,  # Single model input now
                num_steps,
                guidance_scale,
                width,
                height,
                seed
            ],
            outputs=[
                result,
                image_info,
                text_mem,
                gen_mem,
                total_mem,
                text_time,
                gen_time,
                total_time
            ]
        )

    return demo

# Export the generate_images function at module level
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
    api = FluxAPI()
    return api.generate_images(
        prompt=prompt,
        model=model,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        seed=seed,
        batch_size=batch_size,
        n_iter=n_iter,
        return_pil=return_pil
    )

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

def get_app():
    """Create and configure FastAPI app for testing"""
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
    
    # Create Gradio interface
    demo = create_ui()
    
    # Create and mount API endpoints
    create_api(app)
    
    # Mount Gradio app to FastAPI
    app = gr.mount_gradio_app(app, demo, path="/")
    
    return app

if __name__ == "__main__":
    main() 