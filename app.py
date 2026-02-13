import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from fpdf import FPDF
import datetime
import os
import gdown

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NeuroScan AI | Houbad Douaa", page_icon="🧠")

# CSS Background
st.markdown("""
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://www.publicdomainpictures.net/pictures/310000/velka/mri-brain-scan.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stAppViewContainer"] > .main::before {
    content: "";
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    background-color: rgba(255, 255, 255, 0.85);
    z-index: -1;
}
</style>
""", unsafe_allow_html=True)

# --- 2. SIGNATURE ---
st.markdown(f"""
    <div style="text-align: center; padding: 10px;">
        <h1 style="color: #1E3A5F; font-size: 3em; font-weight: 900; margin-bottom: 0;">HOUBAD DOUAA</h1>
        <p style="color: #4A90E2; font-size: 1.2em;">Intelligence Artificielle & Neuro-Radiologie</p>
        <hr>
    </div>
    """, unsafe_allow_html=True)

# --- 3. CHARGEMENT DU MODÈLE ---
@st.cache_resource
@st.cache_resource
def load_my_model():
    model_path = 'brain_tumor_model_v1.h5'
    file_id = '17S8069HGd_H31pxpMmlFh7eB9TzmTEyE'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        with st.spinner("Téléchargement du modèle IA de Houbad Douaa..."):
            gdown.download(url, model_path, quiet=False)
    
    try:
        # Tentative avec le chargement Keras moderne
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        try:
            # Tentative de chargement en ignorant les erreurs de structure (Mode Legacy)
            import h5py
            return tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
        except Exception as e:
            st.error(f"Le modèle a un conflit de structure : {e}")
            return None
# --- 4. LOGIQUE PRINCIPALE ---
model = load_my_model()
class_names = ['Gliome', 'Méningiome', 'Pas de tumeur', 'Pituitaire']

with st.sidebar:
    st.header("👤 Patient")
    nom = st.text_input("Nom complet")
    age = st.number_input("Âge", 0, 120, 25)
    st.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/douaa-houbad-006b6a305)")

file = st.file_uploader("IRM Cérébrale (JPG/PNG)", type=["jpg", "jpeg", "png"])

if file and model:
    img = Image.open(file).convert('RGB')
    st.image(img, caption="IRM transmise", use_container_width=True)
    
    if st.button("Lancer le Diagnostic"):
        # Prétraitement
        img_resized = img.resize((224, 224))
        img_arr = np.array(img_resized) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)
        
        # Prédiction
        pred = model.predict(img_arr)
        idx = np.argmax(pred)
        conf = np.max(pred) * 100
        diag = class_names[idx]
        
        st.success(f"Résultat : {diag} (Confiance : {conf:.2f}%)")
