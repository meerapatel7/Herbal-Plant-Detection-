"""
app.py - Herbal Plant Identifier (Streamlit)

Put this file, nlp_helper.py and plant_info.json in your Medicinal-plant
folder. The folder 'models' must contain herbal_model.keras and classes.json.

Run with:   python -m streamlit run app.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hide TensorFlow's noisy messages

import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

from nlp_helper import answer_question, DISCLAIMER

# ------------------------------------------------------------------
# 1. Settings and file locations
# ------------------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent
MODEL_PATH   = BASE_DIR / "models" / "herbal_model.keras"
CLASSES_PATH = BASE_DIR / "models" / "classes.json"
INFO_PATH    = BASE_DIR / "plant_info.json"
IMG_SIZE     = (160, 160)      # must match the size used in training

st.set_page_config(page_title="Herbal Plant Identifier", page_icon="🌿")


# ------------------------------------------------------------------
# 2. Loading things (cached, so they load only once)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH))


@st.cache_data
def load_json(path_str):
    with open(path_str, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def predict_top(image_bytes, top_k=3):
    """Return the top_k (plant_name, probability) pairs for a photo."""
    model = load_model()
    class_names = load_json(str(CLASSES_PATH))

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)            # fix phone-photo rotation
    img = img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)

    batch = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    probs = model.predict(batch, verbose=0)[0]

    top = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in top]


# ------------------------------------------------------------------
# 3. Check that every needed file exists
# ------------------------------------------------------------------
missing_files = [p.name for p in (MODEL_PATH, CLASSES_PATH, INFO_PATH) if not p.exists()]
if missing_files:
    st.error("Missing file(s): " + ", ".join(missing_files) +
             ". Check that app.py is inside your Medicinal-plant folder.")
    st.stop()

class_names = load_json(str(CLASSES_PATH))
plant_info  = load_json(str(INFO_PATH))


# ------------------------------------------------------------------
# 4. Page layout
# ------------------------------------------------------------------
st.title("🌿 Herbal Plant Identifier")
st.caption("Upload or capture a photo of a plant to identify it and read "
           "its traditional uses.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Minimum confidence", min_value=0.30, max_value=0.95,
        value=0.60, step=0.05,
        help="If the model is less sure than this, it will not name a plant.")
    st.write(f"The model knows **{len(class_names)}** plants.")
    st.markdown("---")
    st.caption(DISCLAIMER)

source = st.radio("Photo source", ["Upload a photo", "Use camera"], horizontal=True)
if source == "Upload a photo":
    photo = st.file_uploader("Choose a plant photo", type=["jpg", "jpeg", "png"])
else:
    photo = st.camera_input("Take a photo of the plant")


# ------------------------------------------------------------------
# 5. Identify the plant
# ------------------------------------------------------------------
if photo is None:
    st.info("Add a photo to begin.")
    st.stop()

image_bytes = photo.getvalue()
shown = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
st.image(shown, caption="Your photo", use_container_width=True)

with st.spinner("Identifying the plant..."):
    results = predict_top(image_bytes)

best_name, best_conf = results[0]

st.subheader("Top predictions")
for name, conf in results:
    st.write(f"**{name.replace('_', ' ')}**: {conf:.0%}")
    st.progress(min(max(conf, 0.0), 1.0))

selected = None
if best_conf >= threshold:
    selected = best_name
    st.success(f"Detected: **{best_name.replace('_', ' ')}** ({best_conf:.0%} confident)")
else:
    st.warning("The model is not confident enough to name this plant. "
               "Try a clearer, closer photo of a single leaf or flower.")

# let the user correct the model
choice = st.selectbox(
    "Wrong or not sure? Choose the plant yourself:",
    ["(use the detected plant)"] + class_names)
if choice != "(use the detected plant)":
    selected = choice

if selected is None:
    st.stop()


# ------------------------------------------------------------------
# 6. Show the plant's information
# ------------------------------------------------------------------
info = plant_info.get(selected)
st.markdown("---")
st.header(selected.replace("_", " "))

if not info or not info.get("uses"):
    st.info("Information for this plant has not been added yet.")
    st.stop()

text_blob = (info.get("scientific_name", "") + " " + " ".join(info.get("uses", []))).lower()
if "not confirmed" in text_blob or "verify species" in text_blob:
    st.info("Some details for this plant are not fully confirmed yet.")

st.markdown(f"**Scientific name:** *{info['scientific_name']}*")

st.subheader("Traditional uses")
for use in info["uses"]:
    st.markdown(f"- {use}")

st.subheader("How it is used")
st.write(info["how_to_use"])

st.warning("**Precautions:** " + info["precautions"])


# ------------------------------------------------------------------
# 7. Question box (NLP)
# ------------------------------------------------------------------
st.markdown("---")
st.subheader(f"Ask about {selected.replace('_', ' ')}")
question = st.text_input(
    "Type your question",
    placeholder="Is it good for fever? How do I make tea? Is it safe in pregnancy?")

if question:
    answer, kind = answer_question(question, selected, info)
    if kind == "emergency":
        st.error(answer)
    elif kind == "unknown":
        st.info(answer)
    else:
        st.success(answer)

st.caption(DISCLAIMER + " Always consult a doctor or qualified practitioner.")
