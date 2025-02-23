import json
import sys
import time
import base64
import subprocess
import requests
import unittest
from pathlib import Path

# Add parent directory to path to import flux_app
sys.path.append(str(Path(__file__).parent.parent))

# Minimal test parameters
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

def test_connection(base_url="http://127.0.0.1:7860"):
    """Test connection to the Flux API server"""
    # First, test basic connectivity
    endpoints = [
        ("/sdapi/v1/sd-models", "GET", None),
        ("/sdapi/v1/options", "GET", None)
    ]

    print(f"\nTesting connection to Flux API at {base_url}")
    print("=" * 50)

    all_passed = True
    for endpoint, method, payload in endpoints:
        url = f"{base_url.rstrip('/')}{endpoint}"
        print(f"\nTesting {method} {endpoint}...")
        
        try:
            response = requests.request(method, url, timeout=30)
            print(f"✅ Success!")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"❌ Request failed: {str(e)}")
            all_passed = False
        
        time.sleep(1)

    # If basic connectivity works, test image generation
    if all_passed:
        print("\nTesting image generation...")
        url = f"{base_url.rstrip('/')}/sdapi/v1/txt2img"
        
        try:
            print("Sending test image generation request...")
            print("Note: First request might take longer as the model needs to be loaded")
            response = requests.post(url, json=TEST_PARAMS, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                if "images" in result and len(result["images"]) > 0:
                    print("✅ Image generation successful!")
                    
                    # Save the test image
                    image_data = base64.b64decode(result["images"][0].split(",")[1])
                    with open("test_image.png", "wb") as f:
                        f.write(image_data)
                    print("   Test image saved as: test_image.png")
                else:
                    print("❌ No images in response")
                    all_passed = False
            else:
                print(f"❌ Request failed with status code: {response.status_code}")
                all_passed = False
                
        except Exception as e:
            print(f"❌ Image generation failed: {str(e)}")
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The Flux API is accessible.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting tips:")
        print("1. Make sure Flux API is running with: python3.11 flux_app.py")
        print("2. Verify port 7860 is available")
        print("3. Check the Flux API server logs for any errors")
    
    return all_passed

if __name__ == "__main__":
    # Default URL for testing
    api_url = "http://127.0.0.1:7860"
    
    # Allow custom URL from command line
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    
    success = test_connection(api_url)
    sys.exit(0 if success else 1) 