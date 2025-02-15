import io
import base64
import argparse
import gradio as gr
from typing import Optional, List
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import mlx.core as mx
from PIL import Image
import numpy as np
import platform
import sys
from pathlib import Path
import webbrowser
from flux import FluxPipeline

app = FastAPI(title="Flux Image Generator")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

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

def to_latent_size(image_size):
    h, w = image_size
    h = ((h + 15) // 16) * 16
    w = ((w + 15) // 16) * 16

    if (h, w) != image_size:
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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

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

def create_ui(enable_api: bool = False, api_port: int = 7860):
    """Create the Gradio interface"""
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # FLUX Image Generator
            Generate images from text using the FLUX model on Apple Silicon Macs.
            """
        )
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3
                )
                model_type = gr.Radio(
                    choices=["schnell", "dev"],
                    value="schnell",
                    label="Model Type"
                )
                
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
                
                with gr.Row():
                    num_steps = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=None,
                        label="Number of Steps (leave empty for default)"
                    )
                    guidance = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        step=0.5,
                        value=4.0,
                        label="Guidance Scale"
                    )
                
                seed = gr.Number(
                    label="Seed (leave empty for random)",
                    precision=0
                )
                
                with gr.Row():
                    generate_btn = gr.Button("Generate")
                    if enable_api:
                        api_btn = gr.Button("Open API Documentation")
            
            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                save_btn = gr.Button("Save Image", visible=False)
        
        # Handle generation
        generated_image = generate_btn.click(
            fn=generate_for_ui,  # Using the new wrapper function
            inputs=[
                prompt,
                model_type,
                num_steps,
                guidance,
                image_width,
                image_height,
                seed
            ],
            outputs=output_image
        )
        
        # Show save button when image is generated
        generated_image.then(
            lambda: gr.Button(visible=True),
            None,
            save_btn
        )
        
        # Handle API documentation
        if enable_api:
            api_btn.click(
                lambda: webbrowser.open(f"http://127.0.0.1:{api_port}"),
                None,
                None
            )
    
    return demo

def main():
    """Main entry point"""
    try:
        check_system_compatibility()
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="FLUX Image Generator")
        parser.add_argument("--enable-api", action="store_true", help="Enable API endpoint")
        parser.add_argument("--api-port", type=int, default=7860, help="Port for API endpoint")
        args = parser.parse_args()
        
        # Add CORS middleware
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        if args.enable_api:
            # Use uvicorn to run FastAPI
            import uvicorn
            uvicorn.run(
                "flux_app:app",  # Use string reference to app
                host="127.0.0.1",
                port=args.api_port,
                reload=False,
                log_level="info"
            )
        else:
            # Create and launch UI only
            demo = create_ui(enable_api=args.enable_api, api_port=args.api_port)
            demo.launch(
                server_name="127.0.0.1",
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

@app.get("/sdapi/v1/sd-models", include_in_schema=True)
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

@app.get("/")
async def root():
    """Root endpoint for testing"""
    print("Root endpoint called")  # Debug log
    return {"message": "Flux API is running"}

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