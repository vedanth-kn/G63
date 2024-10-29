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
MODEL = tf.keras.models.load_model(r"C:\Users\user\OneDrive\Desktop\Final Year Project\Project\model.keras")
CLASS_NAMES = [
    'Pepper-Bell Bacterial Spot',
    'Pepper-Bell Healthy',
    'Potato Early Blight',
    'Potato Late Blight',
    'Potato Healthy',
    'Tomato Bacterial Spot',
    'Tomato Early Blight',
    'Tomato Late Blight',
    'Tomato Leaf Mold',
    'Tomato Septoria Leaf Spot',
    'Tomato Spider Mites Two Spotted Spider Mite',
    'Tomato Target Spot',
    'Tomato YellowLeaf Curl_Virus',
    'Tomato Mosaic Virus',
    'Tomato Healthy'
]

# Load translations
def load_translations(language: str):
    translations_path = f"locales/{language}.json"
    if not os.path.exists(translations_path):
        translations_path = "locales/en.json"  # Default to English if the requested language isn't available
    with open(translations_path, "r", encoding="utf-8") as file:
        return json.load(file)

def format_class_name(class_name, translations):
    formatted_name = class_name.replace('_', ' ').replace('  ', ' ')
    parts = class_name.split(' ')
    plant_name_key = parts[0]  # Get the plant name
    disease_name_key = ' '.join(parts[1:])  # Get the disease name

    # Translate the plant and disease names if translations are available
    plant_name = translations.get("translations", {}).get(plant_name_key, plant_name_key)
    disease_name = translations.get("translations", {}).get(disease_name_key, disease_name_key)

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
        accept_language: str = Header(default="kn")
):
    # Load the correct language translations
    translations = load_translations(accept_language)

    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Format the class name and get translated information
    plant_name, disease_name = format_class_name(predicted_class, translations)
    disease_info_key = f"{plant_name} {disease_name}"

    # Get disease description and solution
    disease_info = translations.get("diseases", {}).get(disease_info_key, {})
    description = disease_info.get("description", "Information not available")
    solution = disease_info.get("solution", "No solution available")

    return {
        'plant': plant_name,
        'disease': disease_name,
        # 'class': predicted_class,
        'confidence': float(confidence),
        'description': description,
        'solution': solution
    }


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
