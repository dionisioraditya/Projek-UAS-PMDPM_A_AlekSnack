# streamlit_app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import pandas as pd

st.set_page_config(page_title="Prediksi Masakan Nusantara", layout="centered")

st.title("Prediksi Masakan Nusantara (gudeg | gulai | rendang | soto)")
st.write("Model: **ResNet50 (scratch)** — dilatih tanpa pretrained weights.")

# Load Model
MODEL_PATH = "resnet50_best.h5"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model tidak ditemukan: {MODEL_PATH}. Pastikan file .h5 berada di folder yang sama.")
    st.stop()

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(MODEL_PATH)
class_names = ["gudeg", "gulai", "rendang", "soto"]

# Upload & Predict
uploaded_file = st.file_uploader("Upload gambar masakan (jpg/png)", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar input", use_column_width=True)

    # preprocessing
    IMG_SIZE = model.input_shape[1] if model.input_shape and len(model.input_shape) > 1 else 224
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)

    # prediksi
    preds = model.predict(x)[0]
    top_idx = np.argmax(preds)

    st.markdown(f"### 🥘 Prediksi: **{class_names[top_idx]}**")
    st.write(f"**Confidence:** {preds[top_idx]*100:.2f}%")

    # tampilkan 3 teratas
    top3 = np.argsort(preds)[-3:][::-1]
    st.write("Top 3 kemungkinan:")
    for i in top3:
        st.write(f"- {class_names[i]} : {preds[i]*100:.2f}%")

    # tampilkan grafik probabilitas
    df = pd.DataFrame({
        "class": class_names,
        "probability": preds
    }).sort_values(by="probability", ascending=False)
    st.bar_chart(df.set_index("class"))

else:
    st.info("Upload gambar untuk melihat hasil prediksi.")

st.markdown("---")
st.caption("© 2025 Prediksi Masakan Nusantara — Model: ResNet50 dari awal (tanpa pretrained weights)")
