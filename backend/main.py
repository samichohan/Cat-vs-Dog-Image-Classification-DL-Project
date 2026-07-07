import io
import csv
import os
from datetime import datetime

import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import tensorflow as tf

# ---------------- CONFIG ---------------- #
MODEL_PATH = "model/cats_vs_dogs.h5"
MODEL_DOWNLOAD_URL = "https://github.com/samichohan/Cat-vs-Dog-Image-Classification-DL-Project/releases/download/v1.0/cats_vs_dogs.h5"
IMG_SIZE = (150, 150)
MAX_IMAGE_SIZE_MB = 5
LOG_PATH = "data/prediction_logs.csv"

# ---------------- APP INIT ---------------- #
app = FastAPI(
    title="Cat vs Dog Classifier API",
    description="Production API for classifying images as cat or dog using a CNN model.",
    version="1.0.0"
)

# Allow requests from any frontend (Streamlit, browser, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_model_downloaded():
    """Download the model file from GitHub Releases if it isn't present locally."""
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("Model not found locally. Downloading from GitHub Releases...")
    response = requests.get(MODEL_DOWNLOAD_URL, stream=True, timeout=60)
    response.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded successfully.")


# ---------------- LOAD MODEL (once, at startup) ---------------- #
try:
    ensure_model_downloaded()
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"⚠️ Failed to load model at startup: {e}")

# ---------------- REQUEST SCHEMA ---------------- #
class ImageURLRequest(BaseModel):
    url: str


# ---------------- HELPER FUNCTIONS ---------------- #
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize, normalize, and reshape image to match model's expected input."""
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr


def run_prediction(img: Image.Image):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on the server.")

    arr = preprocess_image(img)
    raw_pred = model.predict(arr)[0][0]

    predicted_class = "dog" if raw_pred > 0.5 else "cat"
    confidence = float(raw_pred) if predicted_class == "dog" else float(1 - raw_pred)

    return predicted_class, round(confidence, 4)


def log_prediction(source: str, predicted_class: str, confidence: float):
    """Append each prediction to a CSV log file for monitoring/analytics."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(LOG_PATH)

    with open(LOG_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "source", "predicted_class", "confidence"])
        writer.writerow([datetime.utcnow().isoformat(), source, predicted_class, confidence])


# ---------------- ROUTES ---------------- #
@app.get("/")
def root():
    return {"message": "Cat vs Dog Classifier API is running.", "docs": "/docs"}


@app.get("/health")
def health_check():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_loaded": model is not None
    }


@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported.")

    # Read and validate size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"Image too large ({size_mb:.1f}MB). Max allowed: {MAX_IMAGE_SIZE_MB}MB.")

    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    predicted_class, confidence = run_prediction(img)
    log_prediction(source="upload", predicted_class=predicted_class, confidence=confidence)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }


@app.post("/predict/url")
def predict_url(request: ImageURLRequest):
    try:
        response = requests.get(request.url, timeout=10, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=400, detail="Could not fetch image from the provided URL.")

    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type:
        raise HTTPException(status_code=400, detail="The URL does not point to a valid image.")

    content = response.content
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"Image too large ({size_mb:.1f}MB). Max allowed: {MAX_IMAGE_SIZE_MB}MB.")

    try:
        img = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open the image from the URL.")

    predicted_class, confidence = run_prediction(img)
    log_prediction(source="url", predicted_class=predicted_class, confidence=confidence)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }