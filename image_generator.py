import requests
import io
from PIL import Image
def generate(text):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content
    image_bytes = query({
        "inputs": text,
    })

    image = Image.open(io.BytesIO(image_bytes))
    image.save("./static/images/ig.jpg")

# In a realistic modern style as though taken from a camera, generate a billion dollar interior design