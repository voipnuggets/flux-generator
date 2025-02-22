import unittest
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import json
from unittest.mock import patch, MagicMock
import numpy as np
import gradio as gr

# Add parent directory to path to import flux_app
sys.path.append(str(Path(__file__).parent.parent))
from flux_app import get_app, FluxAPI, SDAPIRequest

class TestAPI(unittest.TestCase):
    """Test cases for API endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client"""
        # Get configured FastAPI app
        app = get_app()
        cls.client = TestClient(app)
    
    def setUp(self):
        """Set up test environment for each test"""
        self.api = FluxAPI()
    
    @patch("flux.FluxPipeline")
    def test_txt2img_endpoint(self, mock_pipeline_class):
        """Test the txt2img endpoint"""
        # Set up mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        
        # Create mock latents and conditioning
        mock_conditioning = np.zeros((1, 64, 64, 4))  # Mock conditioning tensor
        mock_latents = np.zeros((1, 64, 64, 4))  # Mock latents tensor
        
        # Set up the mock pipeline
        mock_pipeline.generate_latents.return_value = iter([mock_conditioning, mock_latents])
        mock_pipeline.decode.return_value = np.zeros((1, 512, 512, 3))  # Mock decoded image
        
        # Create test request
        payload = {
            "prompt": "test image",
            "width": 512,
            "height": 512,
            "steps": 1,
            "cfg_scale": 4.0,
            "batch_size": 1,
            "n_iter": 1,
            "seed": 42,
            "model": "schnell"
        }
        
        with patch.object(FluxAPI, 'init_pipeline', return_value=mock_pipeline):
            response = self.client.post("/sdapi/v1/txt2img", json=payload)
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
        response = self.client.get("/sdapi/v1/progress")
        self.assertEqual(response.status_code, 200)
        
        progress = response.json()
        self.assertIn("progress", progress)
        self.assertIn("state", progress)
        self.assertIn("textinfo", progress)
    
    @patch("flux.FluxPipeline")
    def test_pipeline_reuse(self, mock_pipeline_class):
        """Test that the pipeline is reused for the same model"""
        # Set up mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        
        # Create mock latents and conditioning
        mock_conditioning = np.zeros((1, 64, 64, 4))
        mock_latents = np.zeros((1, 64, 64, 4))
        mock_pipeline.generate_latents.return_value = iter([mock_conditioning, mock_latents])
        mock_pipeline.decode.return_value = np.zeros((1, 512, 512, 3))
        
        # Make two requests with the same model
        payload = {
            "prompt": "test image",
            "model": "schnell",
            "width": 512,
            "height": 512,
            "steps": 1
        }
        
        with patch.object(FluxAPI, 'init_pipeline', return_value=mock_pipeline):
            # First request
            response1 = self.client.post("/sdapi/v1/txt2img", json=payload)
            self.assertEqual(response1.status_code, 200)
            
            # Second request
            response2 = self.client.post("/sdapi/v1/txt2img", json=payload)
            self.assertEqual(response2.status_code, 200)
            
            # Pipeline should be initialized only once
            self.assertEqual(mock_pipeline_class.call_count, 1)

if __name__ == "__main__":
    unittest.main(verbosity=2) 