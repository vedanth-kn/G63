from fastapi import FastAPI, File, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
import json

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
MODEL = tf.keras.models.load_model("../model.keras")
CLASS_NAMES = [
    'Pepper__bell__Bacterial_spot',
    'Pepper__bell__healthy',
    'Potato Early blight',
    'Potato__Late_blight',
    'Potato__Healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Load translations
def load_translations(language: str):
    translations_path = f"locales/{language}.json"
    if not os.path.exists(translations_path):
        translations_path = "locales/en.json"  # Default to English if the requested language isn't available
    with open(translations_path, "r", encoding="utf-8") as file:
        return json.load(file)

def format_class_name(class_name):
    formatted_name = class_name.replace('_', ' ').replace('  ', ' ')
    parts = formatted_name.split(' ')
    plant_name = parts[0]
    disease_name = ' '.join(parts[1:])
    return plant_name, disease_name

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
        accept_language: str = Header(default="en")
):
    # Load the correct language translations
    translations = load_translations(accept_language)

    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Format the class name and get translated information
    plant_name, disease_name = format_class_name(predicted_class)
    disease_info_key = f"{plant_name} {disease_name}"

    # Retrieve plant name, disease name, description, solution, and link from translations
    translated_plant_name = translations.get(disease_info_key, {}).get("plant_name", plant_name)
    translated_disease_name = translations.get(disease_info_key, {}).get("disease_name", disease_name)
    description = translations.get(disease_info_key, {}).get("description", "Information not available")
    solution = translations.get(disease_info_key, {}).get("solution", "No solution available")
    link = translations.get(disease_info_key, {}).get("link", "No link available")

    return {
        'plant': translated_plant_name,
        'disease': translated_disease_name,
        'confidence': float(confidence),
        'description': description,
        'solution': solution,
        'link': link
    }


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
