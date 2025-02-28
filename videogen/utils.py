# Copyright © 2024 Apple Inc.

import os
import mlx.core as mx
import numpy as np
import imageio

def save_video(frames, path, fps=8):
    """Save a sequence of frames as a video file.
    
    Args:
        frames: List of numpy arrays or MLX arrays representing video frames
        path: Output path for the video file
        fps: Frames per second for the output video
    """
    if isinstance(frames[0], mx.array):
        frames = [frame.astype(mx.uint8).tolist() for frame in frames]
    frames = [np.array(frame) for frame in frames]
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps) 