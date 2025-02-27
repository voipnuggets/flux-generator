# Copyright © 2024 Apple Inc.

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

import mlx.core as mx
import mlx.nn as nn
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import logging

from .utils import save_video
from ..musicgen.t5 import T5ForTextEncoding

class VideoGenError(Exception):
    """Base exception for VideoGen errors."""
    pass

class ModelLoadError(VideoGenError):
    """Raised when model loading fails."""
    pass

class GenerationError(VideoGenError):
    """Raised when video generation fails."""
    pass

class VideoGen(nn.Module):
    """Text to Video Generation model based on Wan2.1-T2V architecture."""
    
    def __init__(self, model_path: str = "Wan-AI/Wan2.1-T2V-1.3B"):
        super().__init__()
        self.model_path = model_path
        self._generation_progress: Dict[str, Union[int, float]] = {
            "step": 0,
            "total_steps": 0,
            "memory_used": 0.0
        }
        
        try:
            # Load model config
            config_path = hf_hub_download(
                repo_id=self.model_path,
                filename="config.json",
                force_download=False  # Use cached version if available
            )
            with open(config_path) as f:
                self.config = json.load(f)
            
            # Load model weights
            weights_path = hf_hub_download(
                repo_id=self.model_path,
                filename="diffusion_pytorch_model.safetensors",
                force_download=False
            )
            self.weights = mx.load(weights_path)
            
            # Load VAE
            vae_path = hf_hub_download(
                repo_id=self.model_path,
                filename="Wan2.1_VAE.pth",
                force_download=False
            )
            self.vae = torch.load(vae_path, map_location="cpu")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}") from e
        
        # Initialize components
        self.text_encoder = self._init_text_encoder()
        self.decoder = self._init_decoder()
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.config["num_train_timesteps"],
            beta_start=self.config["beta_start"],
            beta_end=self.config["beta_end"],
            beta_schedule=self.config["beta_schedule"]
        )
    
    @property
    def generation_progress(self) -> Dict[str, Union[int, float]]:
        """Get current generation progress."""
        return self._generation_progress.copy()
    
    def _init_text_encoder(self):
        """Initialize the T5 text encoder."""
        return T5ForTextEncoding(
            model_name=self.config["text_encoder"]["model_name"],
            max_length=self.config["text_encoder"]["max_length"]
        )
    
    def _init_decoder(self):
        """Initialize the UNet decoder."""
        return UNetDecoder(
            in_channels=self.config["unet"]["in_channels"],
            out_channels=self.config["unet"]["out_channels"],
            block_out_channels=self.config["unet"]["block_out_channels"],
            layers_per_block=self.config["unet"]["layers_per_block"],
            num_attention_heads=self.config["unet"]["num_attention_heads"]
        )
    
    def generate(
        self,
        prompt: str,
        num_frames: int = 16,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[mx.array]:
        """Generate a video from text prompt."""
        try:
            if seed is not None:
                mx.random.seed(seed)
            
            # Reset progress
            self._generation_progress = {
                "step": 0,
                "total_steps": num_inference_steps,
                "memory_used": 0.0
            }
            
            # Validate inputs
            if not prompt:
                raise ValueError("Prompt cannot be empty")
            if num_frames < 1:
                raise ValueError("num_frames must be positive")
            if height < 64 or width < 64:
                raise ValueError("height and width must be at least 64")
            if num_inference_steps < 1:
                raise ValueError("num_inference_steps must be positive")
            if guidance_scale < 1.0:
                raise ValueError("guidance_scale must be >= 1.0")
            
            # Encode text prompt
            text_embeds = self.text_encoder(prompt)
            if negative_prompt is not None:
                neg_embeds = self.text_encoder(negative_prompt)
            else:
                neg_embeds = mx.zeros_like(text_embeds)
            
            # Initialize latents
            latents = mx.random.normal(
                (1, 4, num_frames, height//8, width//8)
            )
            
            # Setup diffusion
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            
            # Denoising loop
            for i, t in enumerate(tqdm(timesteps, desc="Generating video")):
                # Update progress
                self._generation_progress["step"] = i + 1
                self._generation_progress["memory_used"] = mx.memory.used()
                
                # Expand latents for classifier-free guidance
                latent_model_input = mx.concatenate([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Get model prediction
                noise_pred = self.decoder(
                    latent_model_input,
                    t,
                    encoder_hidden_states=mx.concatenate([text_embeds, neg_embeds])
                )
                
                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.split(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                
                # Update latents
                latents = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latents
                ).prev_sample
                
                # Free memory
                mx.eval(latents)
                mx.clear_grads()
            
            # Decode latents to pixel space
            video = self.decoder.decode(latents)
            
            # Denormalize and convert to uint8
            video = ((video + 1) * 127.5).clip(0, 255).astype(mx.uint8)
            
            return [video[0, i] for i in range(num_frames)]
            
        except Exception as e:
            raise GenerationError(f"Video generation failed: {str(e)}") from e


class UNetDecoder(nn.Module):
    """U-Net based video decoder."""
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        num_attention_heads: int = 8,
    ):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, block_out_channels[0], 3, padding=1)
        
        # Down blocks
        self.down_blocks = []
        for i in range(len(block_out_channels)):
            for _ in range(layers_per_block):
                block = ResnetBlock3D(
                    in_channels=block_out_channels[i],
                    out_channels=block_out_channels[i],
                    num_attention_heads=num_attention_heads if i > 0 else 0
                )
                self.down_blocks.append(block)
            if i < len(block_out_channels) - 1:
                self.down_blocks.append(
                    Downsample3D(
                        channels=block_out_channels[i],
                        out_channels=block_out_channels[i + 1]
                    )
                )
        
        # Mid blocks
        self.mid_block = ResnetBlock3D(
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            num_attention_heads=num_attention_heads
        )
        
        # Up blocks
        self.up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        for i in range(len(reversed_block_out_channels)):
            for _ in range(layers_per_block + 1):
                block = ResnetBlock3D(
                    in_channels=reversed_block_out_channels[i],
                    out_channels=reversed_block_out_channels[i],
                    num_attention_heads=num_attention_heads if i < len(reversed_block_out_channels) - 1 else 0
                )
                self.up_blocks.append(block)
            if i < len(reversed_block_out_channels) - 1:
                self.up_blocks.append(
                    Upsample3D(
                        channels=reversed_block_out_channels[i],
                        out_channels=reversed_block_out_channels[i + 1]
                    )
                )
        
        self.conv_out = nn.Conv3d(block_out_channels[0], out_channels, 3, padding=1)
        
    def __call__(self, x, t, encoder_hidden_states):
        # Initial convolution
        h = self.conv_in(x)
        
        # Downsampling
        for block in self.down_blocks:
            h = block(h, t, encoder_hidden_states)
        
        # Middle
        h = self.mid_block(h, t, encoder_hidden_states)
        
        # Upsampling
        for block in self.up_blocks:
            h = block(h, t, encoder_hidden_states)
        
        # Final convolution
        return self.conv_out(h)
    
    def decode(self, latents):
        # Scale and decode the latents using VAE
        latents = 1 / 0.18215 * latents
        return self.vae.decode(latents).sample


class DDIMScheduler:
    """DDIM scheduler for diffusion process."""
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear"
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        
        # Initialize betas and alphas
        self.betas = self._get_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas)
        
        # Initialize timesteps
        self.timesteps = None
        
    def _get_betas(self):
        if self.beta_schedule == "scaled_linear":
            return mx.linspace(
                self.beta_start**0.5,
                self.beta_end**0.5,
                self.num_train_timesteps
            ) ** 2
        else:
            raise NotImplementedError(
                f"{self.beta_schedule} is not implemented for {self.__class__}"
            )
    
    def set_timesteps(self, num_inference_steps: int):
        self.timesteps = mx.linspace(
            self.num_train_timesteps - 1,
            0,
            num_inference_steps
        ).astype(mx.int32)
    
    def scale_model_input(self, sample: mx.array, timestep: int) -> mx.array:
        s = self.alphas_cumprod[timestep] ** 0.5
        return sample / s
    
    def step(
        self,
        model_output: mx.array,
        timestep: int,
        sample: mx.array,
    ) -> mx.array:
        # Get alpha values
        alpha = self.alphas_cumprod[timestep]
        alpha_next = self.alphas_cumprod[timestep - 1] if timestep > 0 else mx.array(1.0)
        
        # Get predicted x0
        pred_original_sample = (sample - (1 - alpha).sqrt() * model_output) / alpha.sqrt()
        
        # Get coefficient for predicted x0
        pred_original_sample_coeff = (alpha_next.sqrt() * self.betas[timestep]) / (
            1 - alpha
        )
        
        # Get coefficient for current sample
        current_sample_coeff = self.alphas[timestep].sqrt() * (1 - alpha_next) / (
            1 - alpha
        )
        
        # Compute previous sample
        prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )
        
        return prev_sample 


class ResnetBlock3D(nn.Module):
    """3D ResNet block with optional attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_attention_heads: int = 0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First convolution block
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        
        # Second convolution block
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        
        # Optional attention block
        self.attn = None
        if num_attention_heads > 0:
            self.attn = SpatialTransformer(
                out_channels,
                num_attention_heads,
                out_channels // num_attention_heads
            )
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_connection = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def __call__(self, x, t, encoder_hidden_states=None):
        h = x
        h = self.norm1(h)
        h = nn.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)
        
        if self.attn is not None:
            h = self.attn(h, encoder_hidden_states)
        
        return h + self.skip_connection(x)


class Downsample3D(nn.Module):
    """3D downsampling using strided convolution."""
    
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        
        self.conv = nn.Conv3d(
            self.channels,
            self.out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
    
    def __call__(self, x, *args, **kwargs):
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D upsampling using interpolation and convolution."""
    
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        
        self.conv = nn.Conv3d(
            self.channels,
            self.out_channels,
            kernel_size=3,
            padding=1
        )
    
    def __call__(self, x, *args, **kwargs):
        # Interpolate spatially
        b, c, f, h, w = x.shape
        x = mx.reshape(x, (b * f, c, h, w))
        x = mx.image.resize(x, (h * 2, w * 2), method="nearest")
        x = mx.reshape(x, (b, c, f, h * 2, w * 2))
        
        # Interpolate temporally
        x = mx.transpose(x, (0, 1, 3, 4, 2))  # B, C, H, W, F
        b, c, h, w, f = x.shape
        x = mx.reshape(x, (b * c * h, w, f))
        x = mx.image.resize(x, (w, f * 2), method="nearest")
        x = mx.reshape(x, (b, c, h, w, f * 2))
        x = mx.transpose(x, (0, 1, 4, 2, 3))  # B, C, F, H, W
        
        return self.conv(x)


class SpatialTransformer(nn.Module):
    """Multi-head self-attention transformer block."""
    
    def __init__(
        self,
        channels: int,
        num_heads: int,
        head_dim: int
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.norm = nn.LayerNorm(channels)
        
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.to_out = nn.Linear(channels, channels)
    
    def __call__(self, x, context=None):
        b, c, f, h, w = x.shape
        
        # Flatten spatial dimensions
        x = mx.transpose(x, (0, 2, 3, 4, 1))  # B, F, H, W, C
        x = mx.reshape(x, (b, f * h * w, c))
        
        # Apply layer norm
        x = self.norm(x)
        
        # Project to q, k, v
        qkv = self.to_qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        
        # Reshape to heads
        q = mx.reshape(q, (b, -1, self.num_heads, self.head_dim))
        k = mx.reshape(k, (b, -1, self.num_heads, self.head_dim))
        v = mx.reshape(v, (b, -1, self.num_heads, self.head_dim))
        
        # Compute attention
        attn = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = mx.softmax(attn, axis=-1)
        
        out = mx.matmul(attn, v)
        out = mx.reshape(out, (b, -1, self.channels))
        
        # Project back to channels
        out = self.to_out(out)
        
        # Reshape back to video tensor
        out = mx.reshape(out, (b, f, h, w, c))
        out = mx.transpose(out, (0, 4, 1, 2, 3))  # B, C, F, H, W
        
        return out 