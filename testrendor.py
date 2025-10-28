import requests

url = "https://plantdiseaseapi.onrender.com/predict"
image_path = "pep.jpg"

with open(image_path, "rb") as img:
    files = {"image": img}
    response = requests.post(url, files=files)

print("✅ Prediction Result:")
print(response.json())