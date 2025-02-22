import requests
import json
import base64
import unittest
from pathlib import Path
import sys
from PIL import Image
import io
import os
from unittest.mock import patch, MagicMock
import numpy as np

# Add parent directory to path to import flux_app
sys.path.append(str(Path(__file__).parent.parent))
from flux_app import generate_images, SDAPIRequest, to_latent_size, FluxAPI

url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
payload = {
    "prompt": "a beautiful moonset over the ocean, highly detailed, 4k",
    "width": 512,
    "height": 512,
    "steps": 1,
    "cfg_scale": 4.0,
    "batch_size": 1,
    "n_iter": 1,
    "seed": 42,
    "model": "schnell"
}

try:
    print("Sending request to Flux API...")
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raise an exception for bad status codes
    
    result = response.json()
    print("\nGeneration successful!")
    print(f"Info: {result['info']}")
    print(f"Parameters used: {json.dumps(result['parameters'], indent=2)}")
    
    # Save the first generated image
    if result["images"]:
        image_data = base64.b64decode(result["images"][0].split(",")[1])
        output_file = "generated_moonset.png"
        with open(output_file, "wb") as f:
            f.write(image_data)
        print(f"\nImage saved as: {output_file}")
    
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the server. Make sure Flux app is running on port 7860")
except Exception as e:
    print(f"Error: {str(e)}")

# Test models endpoint
try:
    print("\nTesting models endpoint...")
    models_response = requests.get("http://127.0.0.1:7860/sdapi/v1/sd-models")
    models_response.raise_for_status()
    print("Available models:", json.dumps(models_response.json(), indent=2))
except Exception as e:
    print(f"Error getting models: {str(e)}")

class TestImageGeneration(unittest.TestCase):
    """Test cases for image generation functionality"""
    
    def test_latent_size_calculation(self):
        """Test that latent size is calculated correctly"""
        # Test standard sizes
        self.assertEqual(to_latent_size((512, 512)), (64, 64))
        self.assertEqual(to_latent_size((768, 512)), (96, 64))
        
        # Test non-standard sizes
        self.assertEqual(to_latent_size((500, 500)), (64, 64))  # Should round up to 512x512
        self.assertEqual(to_latent_size((513, 513)), (66, 66))  # Should round up to 528x528
    
    def test_generation_request_validation(self):
        """Test that generation request validation works"""
        # Test valid request
        request = SDAPIRequest(
            prompt="test image",
            model="schnell",
            width=512,
            height=512,
            steps=1,
            cfg_scale=4.0,
            seed=42,
            batch_size=1,
            n_iter=1
        )
        self.assertEqual(request.model, "schnell")
        self.assertEqual(request.width, 512)
        self.assertEqual(request.height, 512)
        
        # Test default values
        request = SDAPIRequest(prompt="test")
        self.assertEqual(request.model, "schnell")
        self.assertEqual(request.width, 512)
        self.assertEqual(request.height, 512)
        self.assertEqual(request.cfg_scale, 4.0)
        self.assertIsNone(request.steps)
        self.assertEqual(request.seed, -1)
    
    @patch("flux.FluxPipeline")
    def test_generate_images(self, mock_pipeline_class):
        """Test that images can be generated"""
        # Set up mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        
        # Create mock latents and conditioning
        mock_conditioning = np.zeros((1, 64, 64, 4))  # Mock conditioning tensor
        mock_latents = np.zeros((1, 64, 64, 4))  # Mock latents tensor
        
        # Set up the mock pipeline
        mock_pipeline.generate_latents.return_value = iter([mock_conditioning, mock_latents])
        mock_pipeline.decode.return_value = np.zeros((1, 512, 512, 3))  # Mock decoded image
        
        # Create API instance
        api = FluxAPI()
        
        # Generate images
        images = api.generate_images(
            prompt="test image",
            model="schnell",
            width=512,
            height=512,
            steps=1,
            guidance=4.0,
            seed=42,
            batch_size=1,
            n_iter=1
        )
        
        # Verify the pipeline was initialized and used correctly
        mock_pipeline_class.assert_called_once_with("flux-schnell")
        mock_pipeline.generate_latents.assert_called_once()
        mock_pipeline.decode.assert_called_once()
        
        # Verify we got a base64 encoded image back
        self.assertTrue(len(images) > 0)
        self.assertTrue(images[0].startswith("data:image/png;base64,"))
    
    @patch('requests.post')
    def test_generate_with_api(self, mock_post):
        """Test the API image generation function"""
        # Set up test parameters
        prompt = "test image"
        model_type = "schnell"
        num_steps = 1
        guidance_scale = 4.0
        width = 512
        height = 512
        seed = 42
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "images": ["data:image/png;base64,test123"],
            "info": "Generated with Flux schnell model",
            "parameters": {
                "prompt": prompt,
                "model": model_type,
                "steps": num_steps,
                "cfg_scale": guidance_scale,
                "width": width,
                "height": height,
                "seed": seed
            }
        }
        mock_post.return_value = mock_response
        
        # Generate image using the API
        try:
            response = requests.post(url, json={
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": num_steps,
                "cfg_scale": guidance_scale,
                "batch_size": 1,
                "n_iter": 1,
                "seed": seed,
                "model": model_type
            })
            response.raise_for_status()
            result = response.json()
            self.assertIn("images", result)
            self.assertGreater(len(result["images"]), 0)
            self.assertTrue(result["images"][0].startswith("data:image/png;base64,"))
            
            # Verify the request
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            self.assertEqual(args[0], url)
            self.assertEqual(kwargs["json"]["prompt"], prompt)
            self.assertEqual(kwargs["json"]["steps"], num_steps)
            self.assertEqual(kwargs["json"]["cfg_scale"], guidance_scale)
            self.assertEqual(kwargs["json"]["width"], width)
            self.assertEqual(kwargs["json"]["height"], height)
            self.assertEqual(kwargs["json"]["seed"], seed)
            self.assertEqual(kwargs["json"]["model"], model_type)
            
        except Exception as e:
            self.fail(f"Test failed: {str(e)}")

if __name__ == "__main__":
    unittest.main(verbosity=2) 