import argparse
import gradio as gr
import mlx.core as mx
import mlx.nn as nn
from PIL import Image
import numpy as np
import platform
import sys
from pathlib import Path
from typing import Optional
import webbrowser
from flux import FluxPipeline

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

def generate_image(
    prompt: str,
    model_type: str = "schnell",
    num_steps: Optional[int] = None,
    guidance: float = 4.0,
    image_width: int = 512,
    image_height: int = 512,
    seed: Optional[int] = None,
) -> tuple[Image.Image, str]:
    """Generate image using FLUX model"""
    # Initialize peak memory trackers
    peak_mem_conditioning = 0
    peak_mem_generation = 0
    peak_mem_decoding = 0
    peak_mem_overall = 0
    
    # Load the model
    flux = FluxPipeline("flux-" + model_type)
    num_steps = num_steps or (50 if model_type == "dev" else 2)
    
    # Set up generation parameters
    latent_size = to_latent_size((image_height, image_width))
    
    # Generate latents
    print("Generating image...")
    latents = flux.generate_latents(
        prompt,
        n_images=1,
        num_steps=num_steps,
        latent_size=latent_size,
        guidance=guidance,
        seed=seed
    )

    # Process conditioning
    conditioning = next(latents)
    mx.eval(conditioning)
    peak_mem_conditioning = mx.metal.get_peak_memory() / 1024**3  # Convert to GB

    # Free up memory
    del flux.t5
    del flux.clip

    # Run denoising steps
    for i, x_t in enumerate(latents):
        print(f"Step {i+1}/{num_steps}")
        mx.eval(x_t)
    peak_mem_generation = mx.metal.get_peak_memory() / 1024**3

    # Free up memory
    del flux.flow

    # Decode the image
    decoded = flux.decode(x_t, latent_size)
    mx.eval(decoded)
    peak_mem_decoding = mx.metal.get_peak_memory() / 1024**3
    peak_mem_overall = max(peak_mem_conditioning, peak_mem_generation, peak_mem_decoding)

    # Convert to PIL Image
    x = (decoded * 255).astype(mx.uint8)
    image = Image.fromarray(np.array(x[0]))
    
    # Create memory report
    memory_report = (
        f"Memory Usage Report:\n"
        f"Text conditioning: {peak_mem_conditioning:.3f}GB\n"
        f"Image generation: {peak_mem_generation:.3f}GB\n"
        f"Image decoding: {peak_mem_decoding:.3f}GB\n"
        f"Peak overall: {peak_mem_overall:.3f}GB"
    )
    
    return image, memory_report

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
                memory_text = gr.Textbox(label="Memory Usage Report", interactive=False)
                save_btn = gr.Button("Save Image", visible=False)
        
        # Handle generation
        generated_output = generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                model_type,
                num_steps,
                guidance,
                image_width,
                image_height,
                seed
            ],
            outputs=[output_image, memory_text]
        )
        
        # Show save button when image is generated
        generated_output.then(
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
        
        # Create and launch UI
        demo = create_ui(enable_api=args.enable_api, api_port=args.api_port)
        
        # Launch with share=False to avoid antivirus issues
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