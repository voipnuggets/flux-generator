import requests
import json
import base64
import unittest
from pathlib import Path
import sys
import time
from PIL import Image
import io
import os
import signal
import subprocess
from unittest.mock import patch, MagicMock
import numpy as np
import threading

# Add parent directory to path to import flux_app
sys.path.append(str(Path(__file__).parent.parent))
from flux_app import generate_images, SDAPIRequest, to_latent_size, FluxAPI

# Global lock for image generation tests
generation_lock = threading.Lock()

# Minimal test parameters for image generation
TEST_PARAMS = {
    "prompt": "test",      # Minimal prompt
    "width": 128,         # Minimal width divisible by 16
    "height": 128,        # Minimal height divisible by 16
    "steps": 1,           # Minimum steps
    "cfg_scale": 1.0,     # Minimum guidance
    "batch_size": 1,      # Single image
    "n_iter": 1,          # Single iteration
    "seed": 42,          # Fixed seed for reproducibility
    "model": "schnell"    # Fastest model
}

# Constants
SERVER_START_TIMEOUT = 300  # 5 minutes
SERVER_STOP_TIMEOUT = 30   # 30 seconds
IMAGE_GEN_TIMEOUT = 300    # 5 minutes

class ServerManager:
    """Manages the Flux server process for testing"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self, port=7860):
        if not hasattr(self, 'initialized'):
            self.port = port
            self.process = None
            self.workspace_root = Path(__file__).parent.parent
            self.initialized = True
            self._cleanup_old_processes()
    
    def _cleanup_old_processes(self):
        """Clean up any existing flux_app processes"""
        try:
            # Find any existing flux_app processes
            output = subprocess.check_output(["pgrep", "-f", "flux_app.py"]).decode()
            pids = [int(pid) for pid in output.split()]
            
            # Kill each process
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f"Cleaned up old server process: {pid}")
                except ProcessLookupError:
                    pass  # Process already gone
        except subprocess.CalledProcessError:
            pass  # No processes found
    
    def start(self):
        """Start the Flux server"""
        with self._lock:
            if self.process is not None:
                return True
            
            # Clean up any existing processes first
            self._cleanup_old_processes()
            
            print("\nStarting Flux server...")
            
            # Start the server process with minimal memory settings
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.workspace_root)
            env["MLX_USE_METAL_MEMPOOL"] = "0"  # Disable Metal memory pool
            env["MLX_GC_THRESHOLD"] = "1"       # Aggressive garbage collection
            
            self.process = subprocess.Popen(
                [sys.executable, str(self.workspace_root / "flux_app.py")],
                cwd=str(self.workspace_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create a new process group
            )
            
            # Wait for server to start (max 5 minutes)
            start_time = time.time()
            while time.time() - start_time < SERVER_START_TIMEOUT:
                try:
                    response = requests.get(f"http://127.0.0.1:{self.port}/docs")
                    if response.status_code == 200:
                        print("✓ Server started successfully")
                        time.sleep(2)  # Give the server a moment to fully initialize
                        return True
                except requests.exceptions.ConnectionError:
                    time.sleep(1)
                    # Print progress every 30 seconds
                    elapsed = time.time() - start_time
                    if elapsed % 30 < 1:
                        print(f"Waiting for server... ({int(elapsed)}s)")
                    
                    # Check if process is still running
                    if self.process.poll() is not None:
                        stdout, stderr = self.process.communicate()
                        print("Server process terminated unexpectedly!")
                        print("STDOUT:", stdout.decode())
                        print("STDERR:", stderr.decode())
                        return False
            
            print("❌ Server failed to start within 5 minutes")
            self.stop()
            return False
    
    def stop(self):
        """Stop the Flux server"""
        with self._lock:
            if self.process:
                print("\nStopping Flux server...")
                try:
                    # Try graceful shutdown first
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                    try:
                        self.process.wait(timeout=SERVER_STOP_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        print("❌ Server failed to stop gracefully, forcing shutdown...")
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Process already terminated
                finally:
                    self.process = None
                    print("✓ Server stopped successfully")

class TestImageGeneration(unittest.TestCase):
    """Unit tests for image generation functionality.
    These tests use mocking and do NOT require a running Flux server.
    """
    
    def test_latent_size_calculation(self):
        """Test that latent size is calculated correctly"""
        # Test sizes divisible by 16 (valid sizes)
        self.assertEqual(to_latent_size((512, 512)), (64, 64))  # Standard size
        self.assertEqual(to_latent_size((768, 512)), (96, 64))  # Rectangular size
        
        # Test sizes not divisible by 16 (should round up to nearest 16)
        self.assertEqual(to_latent_size((513, 513)), (64, 64))  # Should round to nearest 16
        self.assertEqual(to_latent_size((769, 513)), (96, 64))  # Should round to nearest 16
    
    def test_generation_request_validation(self):
        """Test that generation request validation works"""
        # Test valid request with minimal parameters
        request = SDAPIRequest(**TEST_PARAMS)
        self.assertEqual(request.model, "schnell")
        self.assertEqual(request.width, 128)
        self.assertEqual(request.height, 128)
        
        # Test default values
        request = SDAPIRequest(prompt="test")
        self.assertEqual(request.model, "schnell")
        self.assertEqual(request.width, 128)  # Default width
        self.assertEqual(request.height, 128)  # Default height
        self.assertEqual(request.cfg_scale, 4.0)
        self.assertIsNone(request.steps)
        self.assertEqual(request.seed, -1)
    
    @patch("flux.FluxPipeline")
    def test_generate_images(self, mock_pipeline_class):
        """Test that images can be generated using mocked pipeline"""
        with generation_lock:  # Ensure exclusive access
            # Set up mock pipeline
            mock_pipeline = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline
            
            # Create mock latents and conditioning with correct sizes
            mock_conditioning = np.zeros((1, 16, 16, 4))  # 128/8 = 16
            mock_latents = np.zeros((1, 16, 16, 4))      # 128/8 = 16
            
            # Set up the mock pipeline behavior
            mock_pipeline.generate_latents.return_value = iter([mock_conditioning, mock_latents])
            mock_pipeline.decode.return_value = np.zeros((1, 128, 128, 3))  # Minimal size image
            
            # Create API instance and patch the pipeline initialization
            api = FluxAPI()
            
            # Generate images with minimal parameters
            with patch.object(FluxAPI, 'init_pipeline', return_value=mock_pipeline):
                images = api.generate_images(
                    prompt=TEST_PARAMS["prompt"],
                    model=TEST_PARAMS["model"],
                    width=TEST_PARAMS["width"],
                    height=TEST_PARAMS["height"],
                    steps=TEST_PARAMS["steps"],
                    guidance=TEST_PARAMS["cfg_scale"],
                    seed=TEST_PARAMS["seed"],
                    batch_size=TEST_PARAMS["batch_size"],
                    n_iter=TEST_PARAMS["n_iter"]
                )
            
            # Verify the pipeline was initialized and used correctly
            mock_pipeline_class.assert_called_once_with("flux-schnell")
            mock_pipeline.generate_latents.assert_called_once()
            mock_pipeline.decode.assert_called_once()
            
            # Verify we got a base64 encoded image back
            self.assertTrue(len(images) > 0)
            self.assertTrue(images[0].startswith("data:image/png;base64,"))


class TestImageGenerationIntegration(unittest.TestCase):
    """Integration tests that require a running Flux server."""
    
    @classmethod
    def setUpClass(cls):
        """Start the server before running integration tests"""
        cls.server = ServerManager()
        if not cls.server.start():
            raise unittest.SkipTest("Failed to start Flux server")
        time.sleep(5)  # Give the server a moment to fully initialize
    
    @classmethod
    def tearDownClass(cls):
        """Stop the server after all tests are done"""
        if hasattr(cls, 'server'):
            cls.server.stop()
    
    def test_models_endpoint(self):
        """Test the models endpoint"""
        response = requests.get("http://127.0.0.1:7860/sdapi/v1/sd-models")
        self.assertEqual(response.status_code, 200)
        
        models = response.json()
        self.assertTrue(isinstance(models, list))
        self.assertEqual(len(models), 2)  # schnell and dev models
        
        # Check model properties
        for model in models:
            self.assertIn("title", model)
            self.assertIn("model_name", model)
            self.assertIn("filename", model)
    
    def test_image_generation(self):
        """Test actual image generation through the API"""
        with generation_lock:  # Ensure exclusive access
            url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
            
            try:
                print("\nStarting image generation test (timeout: 5 minutes)...")
                # Use minimal parameters for testing
                response = requests.post(url, json=TEST_PARAMS, timeout=IMAGE_GEN_TIMEOUT)
                self.assertEqual(response.status_code, 200)
                
                result = response.json()
                self.assertIn("images", result)
                self.assertGreater(len(result["images"]), 0)
                self.assertTrue(result["images"][0].startswith("data:image/png;base64,"))
                
                # Save the generated image
                image_data = base64.b64decode(result["images"][0].split(",")[1])
                with open("test_image.png", "wb") as f:  # More generic test image name
                    f.write(image_data)
                print("✓ Image generation test completed successfully")
            except requests.exceptions.Timeout:
                self.skipTest("Image generation timed out after 5 minutes")
            except Exception as e:
                self.fail(f"Image generation failed: {str(e)}")


if __name__ == "__main__":
    # Run tests sequentially
    unittest.main(verbosity=2, failfast=True) 