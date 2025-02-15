import unittest
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import json

# Add parent directory to path to import flux_app
sys.path.append(str(Path(__file__).parent.parent))
from flux_app import app

class TestAPI(unittest.TestCase):
    """Test cases for API endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client"""
        cls.client = TestClient(app)
    
    def test_txt2img_endpoint(self):
        """Test the txt2img endpoint"""
        payload = {
            "prompt": "test image",
            "width": 512,
            "height": 512,
            "steps": 2,
            "cfg_scale": 4.0,
            "batch_size": 1,
            "n_iter": 1,
            "seed": 42,
            "model": "schnell"
        }
        
        response = self.client.post("/sdapi/v1/txt2img", json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("images", data)
        self.assertIn("parameters", data)
        self.assertIn("info", data)
    
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
    
    def test_cmd_flags_endpoint(self):
        """Test the command flags endpoint"""
        response = self.client.get("/sdapi/v1/cmd-flags")
        self.assertEqual(response.status_code, 200)
        
        flags = response.json()
        self.assertIn("api", flags)
        self.assertIn("ckpt", flags)
        self.assertTrue(flags["api"])

if __name__ == "__main__":
    unittest.main(verbosity=2) 