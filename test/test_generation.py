import requests
import json
import base64
import unittest
from pathlib import Path
import sys
from PIL import Image
import io

# Add parent directory to path to import flux_app
sys.path.append(str(Path(__file__).parent.parent))
from flux_app import generate_images, GenerationRequest, to_latent_size

url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
payload = {
    "prompt": "a beautiful moonset over the ocean, highly detailed, 4k",
    "width": 512,
    "height": 512,
    "steps": 2,
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
        request = GenerationRequest(
            prompt="test image",
            model="schnell",
            n_images=1,
            image_size="512x512",
            steps=2,
            guidance=4.0,
            seed=42
        )
        self.assertEqual(request.model, "schnell")
        self.assertEqual(request.n_images, 1)
        
        # Test default values
        request = GenerationRequest(prompt="test")
        self.assertEqual(request.model, "schnell")
        self.assertEqual(request.n_images, 1)
        self.assertEqual(request.image_size, "512x512")
        self.assertEqual(request.guidance, 4.0)
        self.assertIsNone(request.steps)
        self.assertIsNone(request.seed)
    
    def test_generate_images(self):
        """Test that images can be generated"""
        request = GenerationRequest(
            prompt="test image",
            model="schnell",
            n_images=1,
            image_size="512x512",
            steps=2,
            guidance=4.0,
            seed=42
        )
        
        images = generate_images(request)
        self.assertEqual(len(images), 1)
        
        # Verify image format
        for img_str in images:
            self.assertTrue(img_str.startswith("data:image/png;base64,"))
            
            # Try to decode and open the image
            img_data = base64.b64decode(img_str.split(",")[1])
            img = Image.open(io.BytesIO(img_data))
            self.assertEqual(img.size, (512, 512))
            self.assertEqual(img.mode, "RGB")

if __name__ == "__main__":
    unittest.main(verbosity=2) 