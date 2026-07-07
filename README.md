# 🐾 Cat vs Dog Classifier

A production-style deep learning application that classifies images as **cat** or **dog** using a Convolutional Neural Network (CNN). Built with a decoupled architecture — a **FastAPI** backend serving the model via REST API, and a **Streamlit** frontend consuming it — allowing the model to be used independently of any single UI.

🔗 **Live Demo:** [Add your Streamlit app link here]
🔗 **API Docs:** [cat-vs-dog-image-classification-dl-project-production.up.railway.app]

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [API Reference](#api-reference)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Future Improvements](#future-improvements)

---

## 🔍 Overview

This project implements an end-to-end image classification pipeline using a custom-built CNN trained on the Cats vs Dogs dataset. Unlike a monolithic Streamlit-only app, this project separates concerns into two independently deployable services:

- A **backend API** that owns the model, handles preprocessing, validation, and inference
- A **frontend UI** that only consumes the API — meaning the same backend could serve a mobile app, a browser extension, or any other client without modification

This mirrors how real-world ML systems are typically deployed in production environments.

---

## 🏗 Architecture

```
┌────────────────────┐        HTTP/REST        ┌──────────────────────┐
│  Streamlit Frontend │  ───────────────────►   │   FastAPI Backend     │
│  (Streamlit Cloud)  │  ◄───────────────────   │   (Render/Railway)    │
└────────────────────┘        JSON Response      └──────────────────────┘
                                                          │
                                                          ▼
                                                  ┌─────────────────┐
                                                  │  CNN Model (.h5) │
                                                  └─────────────────┘
```

The frontend supports two input methods — direct image upload and image URL — both of which are sent to the backend for inference and returned as a JSON response containing the predicted class and confidence score.

---

## ✨ Features

- Image classification via **file upload** or **image URL**
- REST API with interactive **Swagger documentation** (`/docs`)
- Backend health check endpoint for monitoring
- Input validation — file type, file size, and corrupt/invalid image handling
- Automatic **prediction logging** to CSV for basic analytics
- CORS-enabled API, allowing any frontend to consume it
- Independently deployable backend and frontend services

---

## 🛠 Tech Stack

| Category            | Tools/Libraries                          |
|---------------------|-------------------------------------------|
| Language            | Python                                    |
| Deep Learning       | TensorFlow / Keras                        |
| Backend API         | FastAPI, Uvicorn                          |
| Frontend            | Streamlit                                 |
| Image Processing    | Pillow, NumPy                             |
| Data Augmentation   | Keras `ImageDataGenerator`                |
| Development         | Jupyter Notebook / Google Colab           |
| Deployment          | Render (backend), Streamlit Cloud (frontend) |

---

## 📂 Project Structure

```
Cat-Dog-Classifier/
│
├── backend/
│   ├── main.py                  # FastAPI application (routes, inference, logging)
│   ├── requirements.txt         # Backend dependencies (FastAPI + TensorFlow)
│   ├── model/
│   │   └── cats_vs_dogs.h5      # Trained CNN model
│   └── data/
│       ├── class_labels.json    # Class index → label mapping
│       └── prediction_logs.csv  # Auto-generated prediction history
│
├── frontend/
│   ├── app.py                   # Streamlit UI
│   └── requirements.txt         # Frontend dependencies (lightweight, no TensorFlow)
│
├── notebook/
│   └── cat_dog_classification.ipynb   # Full data prep, training & evaluation notebook
│
├── .gitignore
└── README.md
```

---

## 📊 Dataset

The model was trained on the **Cats vs Dogs** dataset (via Kaggle), consisting of:

| Split       | Cats  | Dogs  | Total |
|-------------|-------|-------|-------|
| Training    | 4,001 | 4,006 | 8,007 |
| Test        | 1,012 | 1,013 | 2,025 |

Images were augmented during training (rotation, shifting, shearing, zooming, horizontal flipping) to improve generalization, and rescaled to `150x150` pixels with pixel values normalized to `[0, 1]`.

---

## ⚙️ Methodology

1. **Data Loading & Exploration** — Downloaded the dataset via the Kaggle API and inspected class distribution and image dimensions.
2. **Data Augmentation** — Used `ImageDataGenerator` to apply real-time augmentation (rotation, shift, shear, zoom, flip) and a 80/20 train-validation split.
3. **Model Architecture** — Built a custom 3-block CNN:
   - Each block: `Conv2D → MaxPooling2D → BatchNormalization → Dropout`
   - Followed by a `Flatten` layer, a `Dense(512, relu)` layer, and a `Dense(1, sigmoid)` output layer for binary classification.
4. **Training** — Trained for 25 epochs using the Adam optimizer and binary cross-entropy loss.
5. **Evaluation** — Assessed performance on a held-out test set of 2,025 images.
6. **Serialization** — Saved the trained model in HDF5 (`.h5`) format for deployment.
7. **API Development** — Wrapped the model in a FastAPI service with dedicated endpoints for file-based and URL-based predictions, including validation, error handling, and prediction logging.
8. **Frontend Development** — Built a Streamlit interface that communicates with the API over HTTP, supporting both upload and URL-based classification.

---

## 📈 Model Performance

| Metric              | Value   |
|-----------------------|---------|
| Training Accuracy    | ~85.6%  |
| Validation Accuracy   | ~77.6%  |
| **Test Accuracy**     | **76.5%** |
| Test Loss             | 0.576   |

*Note: Performance was achieved with a custom CNN trained from scratch. Transfer learning (e.g., MobileNetV2, EfficientNet) is a planned improvement to push accuracy above 90%.*

---

## 🔌 API Reference

### `GET /health`
Returns backend and model status.
```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict/upload`
Accepts a multipart file upload (`jpg`/`png`, max 5MB).
```json
{ "predicted_class": "dog", "confidence": 0.91 }
```

### `POST /predict/url`
Accepts a JSON body with an image URL.
```json
// Request
{ "url": "https://example.com/image.jpg" }

// Response
{ "predicted_class": "cat", "confidence": 0.87 }
```

Full interactive API documentation is available at `/docs` once the backend is running (Swagger UI, auto-generated by FastAPI).

---

## 🚀 Installation

### Backend

```bash
git clone https://github.com/samichohan/Cat-Dog-Classifier.git
cd Cat-Dog-Classifier/backend

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs at `http://localhost:8000` (interactive docs at `http://localhost:8000/docs`).

### Frontend

```bash
cd ../frontend

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
streamlit run app.py
```

Frontend runs at `http://localhost:8501` and connects to the backend at `http://localhost:8000` by default.

---

## ▶️ Usage

1. Start the backend and frontend as described above.
2. Open the Streamlit app in your browser.
3. Choose either **Upload Image** or **Image URL** tab.
4. Provide an image and click **Classify Image**.
5. View the predicted class and confidence score.

---

## ☁️ Deployment

This project uses a **two-service deployment**:

| Service   | Platform          | Notes                                            |
|-----------|-------------------|---------------------------------------------------|
| Backend   | Render / Railway   | Deployed as a Python web service running Uvicorn  |
| Frontend  | Streamlit Cloud    | `BACKEND_URL` set via Streamlit Secrets to point to the deployed backend |

Once the backend is deployed, its public URL is added to the frontend's Streamlit Cloud secrets:
```toml
BACKEND_URL = "https://your-backend-url.onrender.com"
```

---

## 🔮 Future Improvements

- Apply transfer learning (MobileNetV2/EfficientNet) to improve accuracy beyond 90%
- Convert model to TensorFlow Lite for faster, lighter inference
- Add Grad-CAM visualizations to explain model predictions
- Containerize the backend with Docker for consistent deployment
- Add automated tests for API endpoints
- Add rate limiting and authentication for production-grade API security

---

## 👤 Author

**Sami Chohan**
GitHub: [@samichohan](https://github.com/samichohan)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
