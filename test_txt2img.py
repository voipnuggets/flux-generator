import requests
import json
import base64

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