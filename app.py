import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from fpdf import FPDF
import datetime
import os
import gdown

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NeuroScan AI | Houbad Douaa", page_icon="🧠", layout="wide")

# Style CSS pour le background MRI
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://www.publicdomainpictures.net/pictures/310000/velka/mri-brain-scan.jpg");
        background-size: cover; background-position: center; background-attachment: fixed;
    }
    [data-testid="stAppViewContainer"] > .main::before {
        content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        background-color: rgba(255, 255, 255, 0.88); z-index: -1;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. EN-TÊTE ---
col_h1, col_h2 = st.columns([2, 1])
with col_h1:
    st.markdown('<h1 style="color: #1E3A5F;">HOUBAD DOUAA</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #4A90E2;">Ingénierie Biomédicale & Data Science</h3>', unsafe_allow_html=True)

with col_h2:
    now = datetime.datetime.now()
    st.markdown(f"""<div style="text-align: right; border: 2px solid #1E3A5F; padding: 10px; border-radius: 10px; background: rgba(255,255,255,0.7);">
        📅 {now.strftime("%d/%m/%Y")}<br>⌚ {now.strftime("%H:%M:%S")}</div>""", unsafe_allow_html=True)

# --- 3. RECONSTRUCTION DE L'ARCHITECTURE EXACTE ---
@st.cache_resource
def load_my_model():
    model_path = 'model.keras'
    file_id = '1yYvHXYlkA2NRK4HGD5ANNDW5__mDP-C0'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)
    
    try:
        # On reconstruit l'architecture qui correspond à tes poids (1280 -> 128 -> 4)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=None
        )
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'), # LA COUCHE MANQUANTE ÉTAIT ICI
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        
        # Chargement des poids
        model.load_weights(model_path)
        return model
    except Exception as e:
        st.error(f"Ajustement de l'architecture nécessaire : {e}")
        return None

model = load_my_model()

# --- 4. INTERFACE ---
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 Infos Patient")
    nom = st.text_input("Nom")
    prenom = st.text_input("Prénom")
    date_n = st.date_input("Date de naissance")
    lieu_n = st.text_input("Lieu de naissance")

with col2:
    st.subheader("🔬 Image IRM")
    file = st.file_uploader("Charger l'image", type=["jpg", "png", "jpeg"])

# --- 5. ANALYSE ---


if file is not None and model is not None:
    img = Image.open(file).convert('RGB')
    st.image(img, width=300, caption="IRM Patient")
    
    if st.button("🧬 ANALYSER L'IRM"):
        # Prétraitement MobileNetV2
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        classes = ['Gliome', 'Méningiome', 'Pas de tumeur', 'Pituitaire']
        res_idx = np.argmax(prediction)
        diag = classes[res_idx]
        conf = np.max(prediction) * 100
        
        st.success(f"Diagnostic : {diag} (Indice de confiance : {conf:.2f}%)")
        
        # Rapport PDF
        if nom and prenom:
            with open("temp.png", "wb") as f:
                f.write(file.getbuffer())
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "RAPPORT MEDICAL - NEUROSCAN AI", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, f"Patient : {nom.upper()} {prenom.capitalize()}", ln=True)
            pdf.cell(0, 10, f"Né(e) le : {date_n} à {lieu_n}", ln=True)
            pdf.cell(0, 10, f"Conclusion : {diag}", ln=True)
            pdf.image("temp.png", x=60, w=80)
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("📥 Télécharger le Rapport", pdf_bytes, f"Rapport_{nom}.pdf")

# --- 6. FOOTER ---
st.markdown("---")
st.markdown('<a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank"><button style="width:100%; background-color:#0077B5; color:white; border:none; padding:10px; border-radius:5px; font-weight:bold; cursor:pointer;">for more information cliquer ici</button></a>', unsafe_allow_html=True)
