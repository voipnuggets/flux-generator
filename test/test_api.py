import unittest
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import json
from unittest.mock import patch, MagicMock
import numpy as np
import gradio as gr
import threading

# Add parent directory to path to import flux_app
sys.path.append(str(Path(__file__).parent.parent))
from flux_app import get_app, FluxAPI, SDAPIRequest

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

class TestAPI(unittest.TestCase):
    """Test cases for API endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client and shared resources"""
        # Get configured FastAPI app
        cls.app = get_app()
        cls.client = TestClient(cls.app)
        cls.api_lock = threading.Lock()  # Lock for API calls
    
    def setUp(self):
        """Set up test environment for each test"""
        self.api = FluxAPI()
        # Create mock latents and conditioning with correct sizes for 128x128 images
        self.mock_conditioning = np.zeros((1, 16, 16, 4))  # 128/8 = 16
        self.mock_latents = np.zeros((1, 16, 16, 4))      # 128/8 = 16
        self.mock_decoded = np.zeros((1, 128, 128, 3))    # Minimal size image
    
    def tearDown(self):
        """Clean up after each test"""
        self.api = None
    
    @patch("flux.FluxPipeline")
    def test_txt2img_endpoint(self, mock_pipeline_class):
        """Test the txt2img endpoint"""
        with self.api_lock:  # Ensure exclusive access
            # Set up mock pipeline
            mock_pipeline = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline
            
            # Set up the mock pipeline behavior
            mock_pipeline.generate_latents.return_value = iter([self.mock_conditioning, self.mock_latents])
            mock_pipeline.decode.return_value = self.mock_decoded
            
            with patch.object(FluxAPI, 'init_pipeline', return_value=mock_pipeline):
                response = self.client.post("/sdapi/v1/txt2img", json=TEST_PARAMS)
                self.assertEqual(response.status_code, 200)
                
                # Verify response format
                data = response.json()
                self.assertIn("images", data)
                self.assertTrue(len(data["images"]) > 0)
                self.assertTrue(data["images"][0].startswith("data:image/png;base64,"))
                
                # Verify the pipeline was used correctly
                mock_pipeline.generate_latents.assert_called_once()
                mock_pipeline.decode.assert_called_once()
    
    def test_models_endpoint(self):
        """Test the models endpoint"""
        with self.api_lock:
            response = self.client.get("/sdapi/v1/sd-models")
            self.assertEqual(response.status_code, 200)
            
            models = response.json()
            self.assertTrue(isinstance(models, list))
            self.assertEqual(len(models), 2)  # schnell and dev models
            
            # Check model properties
            for model in models:
                self.assertIn("title", model)
                self.assertIn("model_name", model)
                self.assertIn("filename", model)
    
    def test_options_endpoint(self):
        """Test the options endpoints"""
        with self.api_lock:
            # Get options
            response = self.client.get("/sdapi/v1/options")
            self.assertEqual(response.status_code, 200)
            
            options = response.json()
            self.assertIn("sd_model_checkpoint", options)
            self.assertIn("sd_backend", options)
            self.assertEqual(options["sd_backend"], "Flux MLX")
            
            # Set options
            response = self.client.post("/sdapi/v1/options", json={"test": "value"})
            self.assertEqual(response.status_code, 200)
    
    def test_progress_endpoint(self):
        """Test the progress endpoint"""
        with self.api_lock:
            response = self.client.get("/sdapi/v1/progress")
            self.assertEqual(response.status_code, 200)
            
            progress = response.json()
            self.assertIn("progress", progress)
            self.assertIn("state", progress)
            self.assertIn("textinfo", progress)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)  # Stop on first failure 