import unittest
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import json
from unittest.mock import patch, MagicMock
import numpy as np

# Add parent directory to path to import flux_app
sys.path.append(str(Path(__file__).parent.parent))
from flux_app import app, create_ui

class TestAPI(unittest.TestCase):
    """Test cases for API endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client"""
        # Initialize FastAPI app with all routes
        from flux_app import app, create_ui
        
        # Create Gradio interface
        demo = create_ui(enable_api=True, api_port=7860)
        
        # Mount Gradio app at root
        app.mount("/", demo.app)
        
        cls.client = TestClient(app)
    
    @patch("flux_app.init_pipeline")
    @patch("flux_app.flux_pipeline")
    def test_txt2img_endpoint(self, mock_pipeline, mock_init_pipeline):
        """Test the txt2img endpoint"""
        # Mock the pipeline initialization and generation
        mock_init_pipeline.return_value = mock_pipeline
        
        # Create mock latents and conditioning
        mock_conditioning = np.zeros((1, 64, 64, 4))  # Mock conditioning tensor
        mock_latents = np.zeros((1, 64, 64, 4))  # Mock latents tensor
        
        # Set up the mock pipeline
        mock_pipeline.generate_latents.return_value = iter([mock_conditioning, mock_latents])
        mock_pipeline.decode.return_value = np.zeros((1, 512, 512, 3))  # Mock decoded image
        
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
        
        response = self.client.post("/sdapi/v1/txt2img", json=payload)
        self.assertEqual(response.status_code, 200)
        
        # Verify response format
        data = response.json()
        self.assertIn("images", data)
        self.assertTrue(len(data["images"]) > 0)
        self.assertTrue(data["images"][0].startswith("data:image/png;base64,"))
        
        # Verify the pipeline was initialized and used correctly
        mock_init_pipeline.assert_called_once_with("schnell", False)
        mock_pipeline.generate_latents.assert_called_once()
        mock_pipeline.decode.assert_called_once()
    
    @patch("flux_app.init_pipeline")
    @patch("flux_app.flux_pipeline")
    def test_txt2img_endpoint_with_gradio(self, mock_pipeline, mock_init_pipeline):
        """Test that txt2img endpoint works correctly when mounted with Gradio"""
        # Mock the pipeline initialization and generation
        mock_init_pipeline.return_value = mock_pipeline
        
        # Create mock latents and conditioning
        mock_conditioning = np.zeros((1, 64, 64, 4))  # Mock conditioning tensor
        mock_latents = np.zeros((1, 64, 64, 4))  # Mock latents tensor
        
        # Set up the mock pipeline
        mock_pipeline.generate_latents.return_value = iter([mock_conditioning, mock_latents])
        mock_pipeline.decode.return_value = np.zeros((1, 512, 512, 3))  # Mock decoded image
        
        # Test the API endpoint
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
        
        response = self.client.post("/sdapi/v1/txt2img", json=payload)
        self.assertEqual(response.status_code, 200)
        
        # Verify response format
        data = response.json()
        self.assertIn("images", data)
        self.assertTrue(len(data["images"]) > 0)
        self.assertTrue(data["images"][0].startswith("data:image/png;base64,"))
        
        # Verify the pipeline was initialized and used correctly
        mock_init_pipeline.assert_called_once_with("schnell", False)
        mock_pipeline.generate_latents.assert_called_once()
        mock_pipeline.decode.assert_called_once()
        
        # Verify that Gradio UI is accessible at root
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"].lower())
        
        # Verify that API documentation is accessible
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"].lower())
    
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
    
    def test_manifest_endpoint(self):
        """Test that the web app manifest is accessible"""
        response = self.client.get("/manifest.json")
        self.assertEqual(response.status_code, 200)
        
        manifest = response.json()
        self.assertEqual(manifest["name"], "Flux Image Generator")
        self.assertEqual(manifest["short_name"], "Flux")
        self.assertEqual(manifest["start_url"], "/")
        self.assertEqual(manifest["display"], "standalone")
    
    def test_generate_button_javascript(self):
        """Test that the generate button uses the API directly"""
        # Get the UI page
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        
        # Extract the Gradio configuration
        js_code = response.text
        config_start = js_code.find("window.gradio_config = ") + len("window.gradio_config = ")
        config_end = js_code.find(";</script>", config_start)
        config = json.loads(js_code[config_start:config_end])
        
        # Find the generate button click handler
        for dep in config["dependencies"]:
            if dep.get("api_name") == "on_generate" and dep.get("js"):
                js_code = dep["js"]
                self.assertIn("fetch('/sdapi/v1/txt2img'", js_code)
                self.assertIn("method: 'POST'", js_code)
                self.assertIn("headers: { 'Content-Type': 'application/json' }", js_code)
                break
        else:
            self.fail("Could not find generate button click handler")

if __name__ == "__main__":
    unittest.main(verbosity=2) 