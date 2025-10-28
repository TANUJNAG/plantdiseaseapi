import requests

# API endpoint
url = "http://127.0.0.1:10000/predict"

# Path to your test image
image_path = "pep.jpg"  # Replace with your actual image path

# Send POST request
with open(image_path, "rb") as img_file:
    files = {"image": img_file}
    response = requests.post(url, files=files)

# Print response
if response.status_code == 200:
    result = response.json()
    print("✅ Prediction Result:")
    print(f"Crop Type: {result['crop_type']}")
    print(f"Disease Detected: {result['disease_detected']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Treatment: {result['treatment']}")
else:
    print("❌ Error:", response.status_code)
    print(response.json())