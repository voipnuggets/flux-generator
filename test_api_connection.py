import json
import sys
import time
import base64
import subprocess

def run_curl(url, method="GET", data=None, timeout=30):
    """Run curl command and return response"""
    cmd = ["curl", "-s", "-S", "-v"]  # Added -v for verbose output
    if method == "POST":
        cmd.extend(["-X", "POST", "-H", "Content-Type: application/json"])
        if data:
            cmd.extend(["-d", json.dumps(data)])
    cmd.extend(["-m", str(timeout), url])
    
    print(f"Running curl command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Curl stderr (connection info):\n{result.stderr}")
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Curl error: {e.stderr}")
        raise
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw response: {result.stdout}")
        raise

def test_connection(base_url):
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
            response = run_curl(url, method)
            print(f"✅ Success!")
            print(f"Response: {json.dumps(response, indent=2)}")
        except Exception as e:
            print(f"❌ Request failed: {str(e)}")
            all_passed = False
        
        time.sleep(1)

    # If basic connectivity works, test image generation
    if all_passed:
        print("\nTesting image generation...")
        url = f"{base_url.rstrip('/')}/sdapi/v1/txt2img"
        payload = {
            "prompt": "test",  # Minimal prompt
            "width": 512,
            "height": 512,
            "steps": 2,  # Minimal steps
            "cfg_scale": 4.0,
            "batch_size": 1,
            "n_iter": 1,
            "seed": 42,
            "model": "schnell"
        }
        
        try:
            print("Sending test image generation request (this might take a few minutes)...")
            print("Note: First request might take longer as the model needs to be loaded")
            result = run_curl(url, "POST", payload, timeout=180)  # Increased timeout to 3 minutes
            
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
                
        except Exception as e:
            print(f"❌ Image generation failed: {str(e)}")
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The Flux API is accessible from the container.")
        print("You can now use Open WebUI with the Flux API.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting tips:")
        print("1. Make sure Flux API is running with: python3.11 flux_app.py --enable-api --listen")
        print("2. Verify the host machine's firewall allows incoming connections on port 7860")
        print("3. Check if the container can reach the host using: curl http://host.docker.internal:7860")
        print("4. Check the Flux API server logs for any errors")
        print("5. First image generation might take longer as the model needs to be loaded")
    
    return all_passed

if __name__ == "__main__":
    # Default URL for testing from inside the Open WebUI container
    api_url = "http://host.docker.internal:7860"
    
    # Allow custom URL from command line
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    
    success = test_connection(api_url)
    sys.exit(0 if success else 1) 