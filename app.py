import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from fpdf import FPDF
import datetime
import os
import gdown  # Bibliothèque pour télécharger depuis Drive

# --- 1. CONFIGURATION ET STYLE ---
st.set_page_config(page_title="NeuroScan AI | Houbad Douaa", page_icon="🧠")

# CSS pour le background MRI
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://www.publicdomainpictures.net/pictures/310000/velka/mri-brain-scan.jpg");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}
[data-testid="stAppViewContainer"] > .main::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    background-color: rgba(255, 255, 255, 0.85);
    z-index: -1;
}}
.stApp {{ background: transparent; }}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- 2. SIGNATURE HOUBAD DOUAA ---
st.markdown(f"""
    <div style="text-align: center; padding: 20px;">
        <h3 style="color: #4A90E2; font-family: 'Helvetica'; font-weight: 300; margin-bottom: 0;">PLATEFORME DE DIAGNOSTIC</h3>
        <h1 style="color: #1E3A5F; font-size: 3.5em; font-weight: 900; margin-top: 0; letter-spacing: 2px;">HOUBAD DOUAA</h1>
        <p style="color: #555; font-style: italic;">Intelligence Artificielle appliquée à la Neuro-Radiologie</p>
        <hr style="border: 0; height: 2px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(74, 144, 226, 0.75), rgba(0, 0, 0, 0));">
    </div>
    """, unsafe_allow_html=True)

# --- 3. CHARGEMENT DU MODÈLE DEPUIS DRIVE ---
@st.cache_resource
def load_my_model():
    model_path = 'brain_tumor_model_v1.h5'
    # Ton ID de fichier Drive
    file_id = '17S8069HGd_H31pxpMmlFh7eB9TzmTEyE'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        with st.spinner("Téléchargement du modèle IA de Houbad Douaa (cela peut prendre un moment)..."):
            gdown.download(url, model_path, quiet=False)
    
    return tf.keras.models.load_model(model_path)

# --- 4. FONCTION PDF ---
def generate_pdf(nom, age, diagnostic, confiance):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Plateforme NeuroScan AI - Développée par Houbad Douaa", ln=True, align='R')
    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "RAPPORT D'ANALYSE MÉDICALE", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Date : {datetime.date.today()}", ln=True)
    pdf.cell(200, 10, f"Patient : {nom} ({age} ans)", ln=True)
    pdf.cell(200, 10, f"Diagnostic IA : {diagnostic}", ln=True)
    pdf.cell(200, 10, f"Indice de confiance : {confiance:.2f}%", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# Lancement
model = load_my_model()
class_names = ['Gliome', 'Méningiome', 'Pas de tumeur', 'Pituitaire']

# UI latérale
with st.sidebar:
    st.header("👤 Paramètres")
    nom_patient = st.text_input("Nom du Patient")
    age_patient = st.number_input("Âge", 0, 120, 30)
    st.markdown(f"[🔗 Mon profil LinkedIn](https://www.linkedin.com/in/douaa-houbad-006b6a305)")

# Upload et Analyse
uploaded_file = st.file_uploader("Uploadez l'IRM cérébrale...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="IRM à analyser", use_container_width=True)
    
    if st.button("Lancer l'Analyse"):
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        res_idx = np.argmax(prediction)
        conf = np.max(prediction) * 100
        diag = class_names[res_idx]
        
        st.subheader(f"Résultat : {diag}")
        st.write(f"Confiance : {conf:.2f}%")
        
        if nom_patient:
            pdf_data = generate_pdf(nom_patient, age_patient, diag, conf)
            st.download_button("📥 Télécharger le Rapport PDF", pdf_data, f"Rapport_{nom_patient}.pdf")
