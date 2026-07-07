import streamlit as st
import requests
from PIL import Image
import io

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# ---------------- BACKEND URL ---------------- #
# Set this to your deployed FastAPI backend URL (Render/Railway/HF Spaces) via Streamlit Secrets.
# For local testing (no secrets.toml present), it falls back to localhost.
try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except Exception:
    BACKEND_URL = "http://localhost:8000"

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #2D3142;
        margin-bottom: 0;
    }
    .subtitle {
        color: #6B7280;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background-color: #F8F9FB;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .prediction-label {
        font-size: 2.2rem;
        font-weight: 800;
        text-transform: uppercase;
    }
    .stButton > button {
        background-color: #4C6EF5;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #3B5BDB;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.markdown('<p class="main-title">🐾 Cat vs Dog Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image or paste an image URL to classify it as a cat or a dog.</p>', unsafe_allow_html=True)

# ---------------- BACKEND HEALTH CHECK ---------------- #
def check_backend_health():
    try:
        res = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return res.status_code == 200 and res.json().get("model_loaded", False)
    except Exception:
        return False

backend_healthy = check_backend_health()
if not backend_healthy:
    st.warning("⚠️ Backend API is not reachable right now. Predictions may fail. Please try again shortly.")

# ---------------- TABS: UPLOAD vs URL ---------------- #
tab1, tab2 = st.tabs(["📁 Upload Image", "🔗 Image URL"])

image_to_predict = None
source_label = None

with tab1:
    uploaded_file = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)
        source_label = "upload"
        st.image(image_to_predict, caption="Uploaded Image", use_column_width=True)

with tab2:
    image_url = st.text_input("Paste an image URL", placeholder="https://example.com/image.jpg")
    if image_url:
        try:
            preview_response = requests.get(image_url, timeout=10)
            image_to_predict = Image.open(io.BytesIO(preview_response.content))
            source_label = "url"
            st.image(image_to_predict, caption="Image from URL", use_column_width=True)
        except Exception:
            st.error("Could not load an image from that URL. Please check the link.")

st.divider()

# ---------------- PREDICT BUTTON ---------------- #
if st.button("🔍 Classify Image", disabled=(image_to_predict is None)):
    with st.spinner("Analyzing image..."):
        try:
            if source_label == "upload":
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(f"{BACKEND_URL}/predict/upload", files=files, timeout=30)
            else:
                response = requests.post(f"{BACKEND_URL}/predict/url", json={"url": image_url}, timeout=30)

            if response.status_code == 200:
                result = response.json()
                predicted_class = result["predicted_class"]
                confidence = result["confidence"]

                emoji = "🐶" if predicted_class == "dog" else "🐱"
                color = "#4C6EF5" if predicted_class == "dog" else "#F59E0B"

                st.markdown(
                    f'''
                    <div class="result-card">
                        <div style="font-size: 3rem;">{emoji}</div>
                        <div class="prediction-label" style="color:{color};">{predicted_class}</div>
                        <p style="color:#6B7280;">Confidence: {confidence*100:.1f}%</p>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
                st.progress(confidence)
            else:
                error_detail = response.json().get("detail", "Something went wrong.")
                st.error(f"❌ {error_detail}")

        except requests.exceptions.RequestException:
            st.error("❌ Could not reach the backend API. Please try again later.")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("Built with FastAPI + TensorFlow + Streamlit · Cat vs Dog Classification System")