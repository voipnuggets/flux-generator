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
from musicgen.musicgen import MusicGen
from musicgen.utils import save_audio
import tempfile
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
from videogen import VideoGen, save_video

# Add stable diffusion directory to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "stable_diffusion"))

from stable_diffusion import StableDiffusion, StableDiffusionXL

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
                
                # Save audio to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    save_audio(temp_file.name, audio, model.sampling_rate)
                
                return [
                    temp_file.name,  # Audio output (just the path)
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

def create_video_ui():
    """Create the video generation UI tab."""
    
    # Initialize video model
    video_model = None
    
    with gr.Column():
        gr.Markdown("## 🎬 Text to Video Generation")
        
        with gr.Row():
            with gr.Column(scale=4):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter a description of the video you want to generate...",
                    lines=3
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="What you don't want to see in the video...",
                    lines=2
                )
                
                with gr.Row():
                    with gr.Column():
                        num_frames = gr.Slider(
                            minimum=8,
                            maximum=32,
                            step=8,
                            value=16,
                            label="Number of Frames"
                        )
                        fps = gr.Slider(
                            minimum=4,
                            maximum=30,
                            step=1,
                            value=8,
                            label="FPS"
                        )
                    
                    with gr.Column():
                        width = gr.Slider(
                            minimum=256,
                            maximum=512,
                            step=64,
                            value=256,
                            label="Width"
                        )
                        height = gr.Slider(
                            minimum=256,
                            maximum=512,
                            step=64,
                            value=256,
                            label="Height"
                        )
                
                with gr.Row():
                    with gr.Column():
                        guidance_scale = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            step=0.5,
                            value=7.5,
                            label="Guidance Scale"
                        )
                        num_steps = gr.Slider(
                            minimum=10,
                            maximum=100,
                            step=10,
                            value=50,
                            label="Number of Steps"
                        )
                    
                    with gr.Column():
                        seed = gr.Number(
                            label="Seed (-1 for random)",
                            value=-1,
                            precision=0
                        )
                        generate_button = gr.Button("Generate Video", variant="primary")
            
            with gr.Column(scale=3):
                result = gr.Video(label="Generated Video")
                stats = gr.Textbox(label="Generation Stats", lines=2)
        
        # Example prompts
        gr.Examples(
            examples=[
                ["A beautiful sunset over the ocean waves", "Low quality, blurry", 16],
                ["A blooming flower timelapse", "Dead plants, wilting", 24],
                ["A space journey through colorful nebulas", "Empty space, darkness", 32],
            ],
            inputs=[prompt, negative_prompt, num_frames],
            label="Example Prompts"
        )
        
        def generate_video_wrapper(
            prompt,
            negative_prompt,
            num_frames,
            width,
            height,
            guidance_scale,
            num_steps,
            fps,
            seed
        ):
            try:
                nonlocal video_model
                if video_model is None:
                    video_model = VideoGen()
                
                start_time = time.time()
                start_memory = mx.memory.used()
                
                # Generate video frames
                frames = video_model.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    seed=None if seed < 0 else seed
                )
                
                # Save video to temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                save_video(frames, temp_file.name, fps=fps)
                
                # Calculate stats
                end_time = time.time()
                end_memory = mx.memory.used()
                memory_used = (end_memory - start_memory) / (1024 * 1024 * 1024)  # GB
                generation_time = end_time - start_time
                
                stats_text = f"Generation Time: {generation_time:.2f}s | Memory Used: {memory_used:.2f}GB"
                
                return temp_file.name, stats_text
                
            except Exception as e:
                raise gr.Error(f"Video generation failed: {str(e)}")
        
        generate_button.click(
            fn=generate_video_wrapper,
            inputs=[
                prompt,
                negative_prompt,
                num_frames,
                width,
                height,
                guidance_scale,
                num_steps,
                fps,
                seed
            ],
            outputs=[result, stats]
        )

def create_ui():
    """Create the main UI."""
    with gr.Blocks(title="Flux Generator") as demo:
        gr.Markdown("# Flux Generator")
        
        with gr.Tabs():
            with gr.TabItem("🖼️ Image Generation"):
                create_image_ui()
            with gr.TabItem("🎵 Music Generation"):
                create_musicgen_ui()
            with gr.TabItem("🎬 Video Generation"):
                create_video_ui()
    
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