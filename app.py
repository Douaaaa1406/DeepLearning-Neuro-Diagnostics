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

# CSS Style (MRI Background)
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

# --- 2. EN-TÊTE & HORLOGE ---
col_h1, col_h2 = st.columns([2, 1])
with col_h1:
    st.markdown('<h1 style="color: #1E3A5F;">HOUBAD DOUAA</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #4A90E2;">Ingénierie Biomédicale & Data Science</h3>', unsafe_allow_html=True)

with col_h2:
    now = datetime.datetime.now()
    st.markdown(f"""<div style="text-align: right; border: 2px solid #1E3A5F; padding: 10px; border-radius: 10px;">
        📅 {now.strftime("%d/%m/%Y")}<br>⌚ {now.strftime("%H:%M:%S")}</div>""", unsafe_allow_html=True)

# --- 3. CHARGEMENT SÉCURISÉ DU MODÈLE ---
@st.cache_resource
def load_my_model():
    model_path = 'model.keras'
    file_id = '1yYvHXYlkA2NRK4HGD5ANNDW5__mDP-C0'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)
    
    try:
        # Tentative 1 : Standard
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        try:
            # Tentative 2 : Chargement en tant que couche (plus robuste aux erreurs de shape)
            return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        except Exception as e:
            st.error(f"Erreur critique de structure : {e}")
            return None

# INITIALISATION DE LA VARIABLE (Évite le NameError)
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

# --- 5. LOGIQUE DE DIAGNOSTIC ---
def generate_pdf(nom, prenom, date_n, lieu_n, result, confidence, img_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "RAPPORT MEDICAL NEUROSCAN AI", ln=True, align='C')
    pdf.image(img_path, x=60, w=90)
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Patient : {nom} {prenom}", ln=True)
    pdf.cell(0, 10, f"Diagnostic : {result} ({confidence:.2f}%)", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# VERIFICATION DE SECURITE AVANT PREDICTION
if file is not None:
    if model is None:
        st.error("Le cerveau de l'IA n'est pas prêt. Vérifiez le fichier model.keras.")
    else:
        # On continue seulement si model existe
        img = Image.open(file).convert('RGB')
        st.image(img, width=300)
        
        if st.button("🧬 ANALYSER"):
            with open("temp.png", "wb") as f:
                f.write(file.getbuffer())
            
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array)
            classes = ['Gliome', 'Méningiome', 'Pas de tumeur', 'Pituitaire']
            res_idx = np.argmax(prediction)
            diag = classes[res_idx]
            conf = np.max(prediction) * 100
            
            st.success(f"Résultat : {diag} ({conf:.2f}%)")
            
            if nom and prenom:
                pdf_bytes = generate_pdf(nom, prenom, date_n, lieu_n, diag, conf, "temp.png")
                st.download_button("📥 Télécharger Rapport PDF", pdf_bytes, f"Rapport_{nom}.pdf")

# --- 6. FOOTER ---
st.markdown("---")
st.markdown('<a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank"><button style="width:100%; background-color:#0077B5; color:white; border:none; padding:10px; border-radius:5px; font-weight:bold;">for more information cliquer ici</button></a>', unsafe_allow_html=True)
