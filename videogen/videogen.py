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
import math

from .utils import save_video
from musicgen.t5 import T5

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
        self._generation_progress: Dict[str, int] = {
            "step": 0,
            "total_steps": 0
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
            
            # Set VAE configuration from Wan2.1
            self.vae_stride = (4, 8, 8)  # (temporal, height, width) stride
            self.patch_size = (4, 16, 16)  # (temporal, height, width) patch size
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}") from e
        
        # Initialize components
        self.text_encoder = self._init_text_encoder()
        self.decoder = self._init_decoder()
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.config.get("num_train_timesteps", 1000),
            beta_start=self.config.get("beta_start", 0.00085),
            beta_end=self.config.get("beta_end", 0.012),
            beta_schedule=self.config.get("beta_schedule", "scaled_linear")
        )
        logging.info("VideoGen initialized successfully")
        logging.info(f"Scheduler parameters: steps={self.scheduler.num_train_timesteps}, beta_start={self.scheduler.beta_start}, beta_end={self.scheduler.beta_end}")
    
    @property
    def generation_progress(self) -> Dict[str, int]:
        """Get current generation progress."""
        return self._generation_progress.copy()
    
    def _init_text_encoder(self):
        """Initialize the T5 text encoder."""
        # Use the default T5 model for text encoding
        model, tokenizer = T5.from_pretrained("t5-base")
        self.tokenizer = tokenizer
        return model
    
    def _init_decoder(self):
        """Initialize the UNet decoder."""
        # Default UNet configuration if not found in config
        unet_config = self.config.get("unet", {})
        decoder = UNetDecoder(
            in_channels=unet_config.get("in_channels", 4),
            out_channels=unet_config.get("out_channels", 4),
            block_out_channels=unet_config.get("block_out_channels", (320, 320, 640, 640, 1280, 1280)),
            layers_per_block=unet_config.get("layers_per_block", 2),
            num_attention_heads=unet_config.get("num_attention_heads", 8)
        )
        
        # Load pretrained weights
        decoder_weights = {k: v for k, v in self.weights.items() if k.startswith("decoder")}
        
        # Remove prefix from keys
        fixed_weights = [(k.replace("decoder.", ""), v) for k, v in decoder_weights.items()]
        
        # Log statistics
        logging.info(f"Loading weights for UNetDecoder")
        logging.info(f"Total pretrained weights: {len(fixed_weights)}")
        
        # Use custom weight loading function for more robust handling
        successful, failed = decoder.custom_load_weights(fixed_weights)
        logging.info(f"Weight loading complete: {successful} weights loaded successfully, {failed} weights failed")
        
        # Set VAE for decoding
        decoder.vae = self.vae
        
        return decoder
    
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
        sample_solver: str = "unipc",
    ) -> List[mx.array]:
        """Generate a video from text prompt following Wan2.1 implementation."""
        try:
            if seed is not None:
                mx.random.seed(seed)
                logging.info(f"Using seed: {seed}")
            else:
                import random
                seed = random.randint(0, 2**32 - 1)
                mx.random.seed(seed)
                logging.info(f"Using random seed: {seed}")
            
            # Reset progress
            self._generation_progress = {
                "step": 0,
                "total_steps": num_inference_steps
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
            
            logging.info(f"Generating video with parameters: frames={num_frames}, size={width}x{height}, steps={num_inference_steps}, guidance={guidance_scale}")
            logging.info(f"Prompt: {prompt}")
            if negative_prompt:
                logging.info(f"Negative prompt: {negative_prompt}")
            
            # Encode text prompt
            logging.info("Encoding text prompt...")
            text_embeds = self.text_encoder.encode(self.tokenizer.encode(prompt))
            if negative_prompt is not None:
                neg_embeds = self.text_encoder.encode(self.tokenizer.encode(negative_prompt))
                # Pad to match lengths
                max_length = max(text_embeds.shape[1], neg_embeds.shape[1])
                if text_embeds.shape[1] < max_length:
                    text_embeds = mx.pad(text_embeds, [(0, 0), (0, max_length - text_embeds.shape[1]), (0, 0)])
                if neg_embeds.shape[1] < max_length:
                    neg_embeds = mx.pad(neg_embeds, [(0, 0), (0, max_length - neg_embeds.shape[1]), (0, 0)])
            else:
                neg_embeds = mx.zeros_like(text_embeds)
            
            # Calculate latent dimensions based on VAE stride
            lat_t = (num_frames - 1) // self.vae_stride[0] + 1  # Temporal dimension
            lat_h = height // self.vae_stride[1]  # Height
            lat_w = width // self.vae_stride[2]  # Width
            
            # Calculate sequence length based on patch size
            seq_len = math.ceil(
                (lat_h * lat_w) / (self.patch_size[1] * self.patch_size[2]) * 
                lat_t
            )
            logging.info(f"Calculated sequence length: {seq_len}")
            
            # Initialize latents following Wan2.1 format
            # Shape: [channels, temporal, height, width]
            latents = mx.random.normal(
                (4, lat_t, lat_h, lat_w),
                dtype=mx.float32
            )
            
            logging.info(f"Initialized latents with shape: {latents.shape}")
            
            # Setup diffusion
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            logging.info(f"Set up scheduler with {len(timesteps)} timesteps")
            
            # Denoising loop
            for i, t in enumerate(tqdm(timesteps, desc="Generating video")):
                # Update progress
                self._generation_progress["step"] = i + 1
                
                # Expand latents for classifier-free guidance
                latent_model_input = mx.concatenate([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Prepare time embedding - Wan2.1 expects a specific time embedding format
                t_emb = mx.array([t], dtype=mx.float32)
                
                # Create sinusoidal time embedding of dimension 320 (matching Wan2.1 model)
                half_dim = 160  # half of the time embedding dimension
                emb = mx.log(mx.array(10000.0)) / (half_dim - 1)
                emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb)
                emb = t_emb[:, None] * emb[None, :]
                emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=1)
                t_emb = emb.reshape(1, 320)  # [batch_size, time_embedding_dim]
                
                if i == 0 or i == len(timesteps) - 1:
                    logging.info(f"Time embedding shape at step {i}: {t_emb.shape}")
                
                # Get model prediction
                noise_pred = self.decoder(
                    latent_model_input,
                    t_emb,
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
            
            logging.info("Denoising complete, decoding latents to pixel space...")
            
            try:
                # Scale latents (1 / 0.18215 is the scaling factor used in Stable Diffusion)
                latents = 1 / 0.18215 * latents
                
                # Convert MLX array to PyTorch tensor for VAE
                latents_np = latents.numpy()
                import torch
                latents_torch = torch.from_numpy(latents_np).to(torch.float32)
                
                # Reshape for VAE if needed
                if len(latents_torch.shape) == 4:  # [C, T, H, W]
                    C, T, H, W = latents_torch.shape
                    # Reshape to [B*T, C, H, W] for VAE processing
                    latents_torch = latents_torch.permute(1, 0, 2, 3)  # [T, C, H, W]
                    latents_torch = latents_torch.reshape(T, C, H, W)   # [T, C, H, W]
                
                # Decode through VAE
                with torch.no_grad():
                    video = self.vae.decode(latents_torch).sample
                
                # Convert back to MLX array
                video = mx.array(video.numpy())
                
                # Denormalize and convert to uint8
                video = ((video + 1) * 127.5).clip(0, 255).astype(mx.uint8)
                logging.info(f"Final video shape: {video.shape}")
                
                # Return individual frames
                frames = [video[i] for i in range(video.shape[0])]
                logging.info(f"Generated {len(frames)} frames successfully")
                
                return frames
                
            except Exception as e:
                logging.error(f"Error in final decoding: {str(e)}")
                raise GenerationError(f"Video generation failed during decoding: {str(e)}") from e
            
        except Exception as e:
            logging.error(f"Video generation failed: {str(e)}")
            raise GenerationError(f"Video generation failed: {str(e)}") from e


class CustomConv3D(nn.Module):
    """Custom 3D Convolution layer with proper weight initialization and reshaping."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride, stride)
        self.padding = (padding, padding, padding)
        self.dilation = (1, 1, 1)
        
        # Initialize weights with Xavier/Glorot initialization
        # Shape: [out_channels, in_channels, depth, height, width]
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        scale = 1.0 / (in_channels * kernel_size * kernel_size * kernel_size) ** 0.5
        self.weight = mx.random.normal(weight_shape, scale=scale)
        self.bias = mx.zeros((out_channels,))
        
        logging.info(f"Initialized CustomConv3D with original weight shape: {self.weight.shape}")
    
    def __call__(self, x):
        # Handle both 4D and 5D input tensors
        input_shape = x.shape
        logging.info(f"CustomConv3D input shape: {input_shape}")
        
        if len(input_shape) == 4:  # [batch, channels, height, width]
            b, c, h, w = input_shape
            # Add temporal dimension
            x = mx.expand_dims(x, axis=2)  # [batch, channels, 1, height, width]
            f = 1
        elif len(input_shape) == 5:  # [batch, channels, frames, height, width]
            b, c, f, h, w = input_shape
        else:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {input_shape}")
        
        # Reshape input from [batch, channels, frames, height, width] to [batch * frames, channels, height, width]
        x_reshaped = mx.reshape(x, (b * f, c, h, w))
        
        # Reshape weights from [out_c, in_c, d, h, w] to [out_c, in_c * d, h, w]
        weight_reshaped = mx.reshape(self.weight, 
            (self.out_channels, self.in_channels * self.kernel_size, self.kernel_size, self.kernel_size)
        )
        
        logging.info(f"Reshaped input to: {x_reshaped.shape}")
        logging.info(f"Reshaped weights to: {weight_reshaped.shape}")
        
        # Apply 2D convolution
        y = mx.conv2d(
            x_reshaped, 
            weight_reshaped,
            stride=self.stride[1:],  # Use only spatial strides
            padding=self.padding[1:],  # Use only spatial padding
            dilation=self.dilation[1:]  # Use only spatial dilation
        )
        
        # Add bias
        y = y + self.bias.reshape(-1, 1, 1)
        
        # Get output spatial dimensions
        _, _, h_out, w_out = y.shape
        
        # Reshape back to 5D
        y = mx.reshape(y, (b, f, self.out_channels, h_out, w_out))
        y = mx.transpose(y, (0, 2, 1, 3, 4))  # [b, c, f, h, w]
        
        # If input was 4D, remove the temporal dimension
        if len(input_shape) == 4:
            y = y.squeeze(axis=2)
        
        logging.info(f"CustomConv3D output shape: {y.shape}")
        return y


class UNetDecoder(nn.Module):
    """U-Net based video decoder following Wan2.1 architecture."""
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        block_out_channels: Tuple[int] = (320, 320, 640, 640, 1280, 1280),
        layers_per_block: int = 2,
        num_attention_heads: int = 8,
    ):
        super().__init__()
        logging.info(f"Initializing UNetDecoder with in_channels={in_channels}, out_channels={out_channels}")
        
        # Initialize first convolution with correct weight format from Wan2.1
        self.conv_in = CustomConv3D(
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            kernel_size=3,
            padding=1
        )
        
        # Down blocks
        self.down_blocks = nn.Module()
        in_channel = block_out_channels[0]
        block_idx = 0
        
        for i, out_channel in enumerate(block_out_channels):
            for j in range(layers_per_block):
                block = ResnetBlock3D(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    num_attention_heads=num_attention_heads if i > 0 else 0
                )
                # Use exact naming pattern from Wan2.1: down_blocks.down_blocks.{idx}
                setattr(self.down_blocks, f"down_blocks.{block_idx}", block)
                block_idx += 1
                in_channel = out_channel
            
            if i < len(block_out_channels) - 1:
                downsample = Downsample3D(
                    channels=out_channel,
                    out_channels=block_out_channels[i + 1]
                )
                setattr(self.down_blocks, f"down_blocks.{block_idx}", downsample)
                block_idx += 1
                in_channel = block_out_channels[i + 1]
        
        # Mid block
        self.mid_block = ResnetBlock3D(
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            num_attention_heads=num_attention_heads
        )
        
        # Up blocks
        self.up_blocks = nn.Module()
        reversed_block_out_channels = list(reversed(block_out_channels))
        in_channel = block_out_channels[-1]
        block_idx = 0
        
        for i, out_channel in enumerate(reversed_block_out_channels):
            for j in range(layers_per_block + 1):
                block = ResnetBlock3D(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    num_attention_heads=num_attention_heads if i < len(reversed_block_out_channels) - 1 else 0
                )
                setattr(self.up_blocks, f"up_blocks.{block_idx}", block)
                block_idx += 1
                in_channel = out_channel
            
            if i < len(reversed_block_out_channels) - 1:
                upsample = Upsample3D(
                    channels=out_channel,
                    out_channels=reversed_block_out_channels[i + 1]
                )
                setattr(self.up_blocks, f"up_blocks.{block_idx}", upsample)
                block_idx += 1
                in_channel = reversed_block_out_channels[i + 1]
        
        # Final convolution with correct weight format
        self.conv_out = CustomConv3D(
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
    
    def custom_load_weights(self, weights):
        """Load weights with robust error handling following Wan2.1 format"""
        successful = 0
        failed = 0
        
        # Try loading each weight individually
        for key, weight in weights:
            try:
                # Split key by dots to navigate module hierarchy
                parts = key.split('.')
                target = self
                
                # Navigate to the parent module
                for part in parts[:-1]:
                    if hasattr(target, part):
                        target = getattr(target, part)
                    else:
                        raise AttributeError(f"Module has no attribute '{part}'")
                
                # Get the final attribute name
                final_attr = parts[-1]
                if hasattr(target, final_attr):
                    # Set the parameter value
                    current_param = getattr(target, final_attr)
                    
                    # Handle convolution weight transposition
                    if final_attr == "weight" and isinstance(target, CustomConv3D):
                        # Check if weight dimensions need to be transposed
                        if current_param.shape != weight.shape:
                            logging.info(f"Processing Conv3D weight for {key}")
                            logging.info(f"Target shape: {current_param.shape}, Weight shape: {weight.shape}")
                            
                            # For Conv3D weights, ensure shape is (out_channels, in_channels, kt, kh, kw)
                            if len(weight.shape) == 5:
                                # Try common permutations to match target shape
                                permutations = [
                                    (0, 4, 1, 2, 3),  # (out_c, in_c, t, h, w) from (out_c, t, h, w, in_c)
                                    (0, 1, 2, 3, 4),  # No change
                                    (4, 0, 1, 2, 3),  # (in_c, out_c, t, h, w) to (out_c, in_c, t, h, w)
                                    (0, 3, 1, 2, 4),  # Special case for some weight formats
                                ]
                                
                                found = False
                                for perm in permutations:
                                    try:
                                        test_weight = mx.transpose(weight, perm)
                                        if test_weight.shape == current_param.shape:
                                            weight = test_weight
                                            logging.info(f"Found working permutation: {perm}")
                                            found = True
                                            break
                                    except Exception as e:
                                        continue
                                
                                if not found:
                                    # If no permutation worked, try to reshape the weight tensor
                                    if weight.shape[-1] == current_param.shape[1]:  # If input channels are last
                                        # Move input channels to second position
                                        weight = mx.transpose(weight, (0, 4, 1, 2, 3))
                                        logging.info("Moved input channels from last to second position")
                                    elif weight.shape[1] == current_param.shape[0]:  # If output channels are second
                                        # Move output channels to first position
                                        weight = mx.transpose(weight, (1, 0, 2, 3, 4))
                                        logging.info("Moved output channels from second to first position")
                                    
                                    if weight.shape != current_param.shape:
                                        raise ValueError(
                                            f"Could not find valid permutation for weight shape {weight.shape} "
                                            f"to match {current_param.shape}"
                                        )
                            
                            logging.info(f"Final weight shape: {weight.shape}")
                    
                    # Set the parameter
                    if isinstance(current_param, mx.array):
                        if current_param.shape == weight.shape:
                            setattr(target, final_attr, weight)
                            successful += 1
                            logging.info(f"Successfully loaded weight {key} with shape {weight.shape}")
                        else:
                            logging.error(f"Shape mismatch for {key}: expected {current_param.shape}, got {weight.shape}")
                            failed += 1
                    else:
                        setattr(target, final_attr, weight)
                        successful += 1
                else:
                    logging.warning(f"Missing attribute {final_attr} in target module while loading {key}")
                    failed += 1
            except Exception as e:
                logging.error(f"Error loading weight {key}: {str(e)}")
                failed += 1
        
        logging.info(f"Weight loading complete: {successful} succeeded, {failed} failed")
        return successful, failed
    
    def __call__(self, x, t, encoder_hidden_states):
        # Ensure input has the right shape for Conv3D ops
        # Expected input shape: [batch, channels, frames, height, width]
        if len(x.shape) == 5:
            b, c, f, h, w = x.shape
            logging.info(f"Input shape to UNetDecoder: {x.shape}")
        
        # Log time embedding shape
        logging.info(f"Time embedding shape: {t.shape}")
        
        # Log encoder hidden states shape
        if encoder_hidden_states is not None:
            logging.info(f"Encoder hidden states shape: {encoder_hidden_states.shape}")
        
        # Initial convolution
        h = self.conv_in(x)
        logging.info(f"After conv_in shape: {h.shape}")
        
        # Store residual connections
        residuals = []
        
        # Downsampling
        down_block_names = [name for name in dir(self.down_blocks) if name.startswith("down_blocks.")]
        down_block_names.sort(key=lambda x: int(x.split('.')[-1]))  # Sort by block index
        for name in down_block_names:
            block = getattr(self.down_blocks, name)
            if isinstance(block, ResnetBlock3D):
                residuals.append(h)
            h = block(h, t, encoder_hidden_states)
        
        # Middle
        h = self.mid_block(h, t, encoder_hidden_states)
        
        # Upsampling with skip connections
        up_block_names = [name for name in dir(self.up_blocks) if name.startswith("up_blocks.")]
        up_block_names.sort(key=lambda x: int(x.split('.')[-1]))  # Sort by block index
        for name in up_block_names:
            block = getattr(self.up_blocks, name)
            if isinstance(block, ResnetBlock3D):
                if residuals:
                    h = h + residuals.pop()
            h = block(h, t, encoder_hidden_states)
        
        # Final convolution
        return self.conv_out(h)
    
    def decode(self, latents):
        """Scale and decode the diffusion model output using VAE."""
        logging.info(f"Decoding latents with shape: {latents.shape}")
        
        # Scale latents (1 / 0.18215 is the scaling factor used in the Stable Diffusion model)
        latents = 1 / 0.18215 * latents
        
        # The VAE expects PyTorch tensors, so convert from MLX to PyTorch
        # Convert MLX array to numpy array first
        latents_np = latents.numpy()  
        logging.info(f"Converted latents to numpy array with shape: {latents_np.shape}")
        
        # Convert to PyTorch tensor
        import torch
        latents_torch = torch.from_numpy(latents_np).to(torch.float32)
        
        # For video generation, we need to process each frame separately through the VAE
        if len(latents_torch.shape) == 5:  # [B, C, F, H, W]
            B, C, F, H, W = latents_torch.shape
            logging.info(f"Processing video with {F} frames through VAE")
            
            # Reshape to [B*F, C, H, W] for VAE processing
            latents_torch = latents_torch.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
            latents_torch = latents_torch.reshape(B*F, C, H, W)   # [B*F, C, H, W]
            logging.info(f"Reshaped for VAE processing: {latents_torch.shape}")
            
            # Process through VAE
            with torch.no_grad():
                try:
                    decoded = self.vae.decode(latents_torch).sample
                    logging.info(f"VAE decoded shape: {decoded.shape}")
                except Exception as e:
                    logging.error(f"Error in VAE decoding: {str(e)}")
                    raise
            
            # Reshape back to video format [B, C, F, H, W]
            C_out = decoded.shape[1]
            H_out, W_out = decoded.shape[2], decoded.shape[3]
            decoded = decoded.reshape(B, F, C_out, H_out, W_out)  # [B, F, C, H, W]
            decoded = decoded.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
            logging.info(f"Final decoded video shape: {decoded.shape}")
        else:
            # For single image processing
            logging.info("Processing single image through VAE")
            with torch.no_grad():
                decoded = self.vae.decode(latents_torch).sample
                logging.info(f"Single image decoded shape: {decoded.shape}")
        
        # Convert back to MLX array
        return mx.array(decoded.numpy())


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
        
        # Time embedding projection - Match expected architecture for Wan2.1
        # The model expects a time embedding dimension of 320
        self.time_emb_proj = nn.Sequential([
            nn.SiLU(),
            nn.Linear(320, out_channels)  # 320 is the time embedding dimension in Wan2.1
        ])
        
        # First convolution block
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = CustomConv3D(in_channels, out_channels, 3, padding=1)
        
        # Second convolution block
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(0.1)
        self.conv2 = CustomConv3D(out_channels, out_channels, 3, padding=1)
        
        # Optional attention block
        if num_attention_heads > 0:
            self.attn = SpatialTransformer(
                channels=out_channels,
                num_heads=num_attention_heads,
                head_dim=out_channels // num_attention_heads
            )
            self.attn_norm = nn.GroupNorm(32, out_channels)
        else:
            self.attn = None
            self.attn_norm = None
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_connection = CustomConv3D(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def __call__(self, x, t, encoder_hidden_states=None):
        residual = self.skip_connection(x)
        
        # First conv block
        h = self.norm1(x)
        h = nn.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        if t is not None:
            # Make sure t is a proper embedding (has same batch size as x)
            if len(t.shape) == 1:  # If t is a scalar or 1D array
                t = t.reshape(1, -1)  # Reshape to (1, dim)
            time_emb = self.time_emb_proj(t)
            # Reshape to match spatial dimensions
            time_emb = time_emb.reshape(time_emb.shape[0], time_emb.shape[1], 1, 1, 1)
            h = h + time_emb
        
        # Second conv block
        h = self.norm2(h)
        h = nn.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        h = h + residual
        
        # Optional attention
        if self.attn is not None:
            residual = h
            h = self.attn_norm(h)
            h = self.attn(h, encoder_hidden_states)
            h = h + residual
        
        return h


class Downsample3D(nn.Module):
    """3D downsampling using strided convolution following Wan2.1."""
    
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        
        # Initialize conv with correct weight format from Wan2.1
        self.conv = CustomConv3D(
            in_channels=self.channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
    
    def __call__(self, x, *args, **kwargs):
        # Log input shape for debugging
        if len(x.shape) == 5:
            b, c, f, h, w = x.shape
            logging.info(f"Downsample3D input shape: {x.shape}")
        
        # Apply strided convolution
        x = self.conv(x)
        logging.info(f"Downsample3D output shape: {x.shape}")
        return x


class Upsample3D(nn.Module):
    """3D upsampling using interpolation and convolution following Wan2.1."""
    
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        
        # Initialize conv with correct weight format from Wan2.1
        self.conv = CustomConv3D(
            in_channels=self.channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1
        )
    
    def __call__(self, x, *args, **kwargs):
        # Log input shape for debugging
        if len(x.shape) == 5:
            b, c, f, h, w = x.shape
            logging.info(f"Upsample3D input shape: {x.shape}")
        
        # Interpolate spatially first
        b, c, f, h, w = x.shape
        x = mx.reshape(x, (b * f, c, h, w))
        x = mx.image.resize(x, (h * 2, w * 2), method="nearest")
        x = mx.reshape(x, (b, c, f, h * 2, w * 2))
        
        # Then interpolate temporally
        x = mx.transpose(x, (0, 1, 3, 4, 2))  # [B, C, H, W, F]
        b, c, h, w, f = x.shape
        x = mx.reshape(x, (b * c * h, w, f))
        x = mx.image.resize(x, (w, f * 2), method="nearest")
        x = mx.reshape(x, (b, c, h, w, f * 2))
        x = mx.transpose(x, (0, 1, 4, 2, 3))  # [B, C, F, H, W]
        
        # Apply convolution
        x = self.conv(x)
        logging.info(f"Upsample3D output shape: {x.shape}")
        return x


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
        inner_dim = head_dim * num_heads
        
        # Named components exactly as in pretrained weights
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Linear(channels, inner_dim)
        
        # QKV projections - named as expected by pretrained weights
        self.to_q = nn.Linear(inner_dim, inner_dim)
        self.to_k = nn.Linear(inner_dim, inner_dim)
        self.to_v = nn.Linear(inner_dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, channels)
        
        # Feedforward - named as expected by pretrained weights
        self.ff_norm = nn.LayerNorm(inner_dim)
        self.ff_linear1 = nn.Linear(inner_dim, inner_dim * 4)
        self.ff_linear2 = nn.Linear(inner_dim * 4, inner_dim)
        self.ff_dropout = nn.Dropout(0.1)
    
    def __call__(self, x, context=None):
        b, c, f, h, w = x.shape
        
        # Normalize and flatten
        x = self.norm(x)
        x = mx.transpose(x, (0, 2, 3, 4, 1))  # [b, f, h, w, c]
        x = mx.reshape(x, (b, f * h * w, c))
        
        # Project in
        x = self.proj_in(x)
        
        # Self-attention or cross-attention
        if context is not None and context.shape[0] > 0:
            # If using cross-attention with context
            # Use input as query, context as key/value
            q = self.to_q(x)
            k = self.to_k(context)
            v = self.to_v(context)
        else:
            # Self-attention
            q = self.to_q(x)
            k = self.to_k(x)
            v = self.to_v(x)
        
        # Reshape to heads
        q = mx.reshape(q, (b, -1, self.num_heads, self.head_dim))
        k = mx.reshape(k, (k.shape[0], -1, self.num_heads, self.head_dim))
        v = mx.reshape(v, (v.shape[0], -1, self.num_heads, self.head_dim))
        
        # Transpose for attention
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        
        # Attention
        attn = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = mx.softmax(attn, axis=-1)
        
        # Merge heads
        x = mx.matmul(attn, v)
        x = mx.transpose(x, (0, 2, 1, 3))
        x = mx.reshape(x, (b, -1, self.num_heads * self.head_dim))
        
        # Project out
        x = self.to_out(x)
        
        # Feedforward
        residual = x
        x = self.ff_norm(x)
        x = self.ff_linear1(x)
        x = nn.gelu(x)
        x = self.ff_dropout(x)
        x = self.ff_linear2(x)
        x = x + residual
        
        # Reshape back
        x = mx.reshape(x, (b, f, h, w, c))
        x = mx.transpose(x, (0, 4, 1, 2, 3))  # [b, c, f, h, w]
        
        return x 