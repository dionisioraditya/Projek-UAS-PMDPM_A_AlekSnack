# streamlit_app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from tensorflow.keras import layers, models

st.set_page_config(page_title="Prediksi Masakan Nusantara", layout="centered")

st.title("Prediksi Masakan Nusantara (gudeg | gulai | rendang | soto)")
st.write("Model: **Custom definisi sendiri (scratch)**")

# Load Model
MODEL_PATH = "Custom_A_AlekSnack.weights.h5"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model tidak ditemukan: {MODEL_PATH}. Pastikan file .h5 berada di folder yang sama.")
    st.stop()

def build_aleksnack_model(input_shape=(224,224,3), num_classes=4):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')
    ])
    return model

@st.cache_resource
def load_model_from_weights(path):
    model = build_aleksnack_model(input_shape=(224,224,3), num_classes=4)
    model.load_weights(path)
    return model

model = load_model_from_weights(MODEL_PATH)
class_names = ["gudeg", "gulai", "rendang", "soto"]

# Upload & Predict
uploaded_file = st.file_uploader("Upload gambar makanan rendang/ soto/ gulai/ gudeg (jpg/png)", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar input", use_column_width=True)

    # preprocessing sesuai notebook: x/255.0 dan ukuran 224×224
    img_resized = img.resize((224, 224))
    x = np.array(img_resized, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)[0]
    top_idx = np.argmax(preds)

    st.markdown(f"### 🥘 Prediksi: **{class_names[top_idx]}**")
    st.write(f"**Confidence:** {preds[top_idx]*100:.2f}%")

    df = pd.DataFrame({"class": class_names, "probability": preds}).sort_values("probability", ascending=False)
    st.bar_chart(df.set_index("class"))

else:
    st.info("Upload gambar untuk melihat hasil prediksi.")

st.markdown("---")
st.caption("© 2025 Prediksi Masakan Nusantara — Model: ResNet50 dari awal (tanpa pretrained weights)")
