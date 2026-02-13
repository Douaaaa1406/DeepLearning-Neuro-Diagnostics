import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from fpdf import FPDF
import datetime
import os
import gdown
import time

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="NeuroScan AI | Houbad Douaa", page_icon="🧠", layout="wide")

# --- 2. STYLE CSS AVANCÉ (Background MRI & Transparence) ---
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
        background-color: rgba(255, 255, 255, 0.90); /* Voile blanc professionnel */
        z-index: -1;
    }
    .stTextInput, .stNumberInput, .stDateInput {
        background-color: rgba(255, 255, 255, 0.7);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. HORLOGE TEMPS RÉEL (Haut Droite) ---
col_header_1, col_header_2 = st.columns([2, 1])
with col_header_2:
    now = datetime.datetime.now()
    st.markdown(f"""
        <div style="text-align: right; color: #1E3A5F; font-weight: bold; padding: 10px; border: 1px solid #4A90E2; border-radius: 5px;">
            📅 {now.strftime("%d/%m/%Y")}<br>
            ⌚ {now.strftime("%H:%M:%S")}
        </div>
    """, unsafe_allow_html=True)

# --- 4. SIGNATURE & TITRE ---
with col_header_1:
    st.markdown("""
        <h1 style="color: #1E3A5F; margin-bottom: 0;">PLATEFORME NEUROSCAN AI</h1>
        <h3 style="color: #4A90E2; font-weight: 300; margin-top: 0;">Ingénierie Biomédicale & Data Science | Houbad Douaa</h3>
    """, unsafe_allow_html=True)

# --- 5. CHARGEMENT DU MODÈLE (.keras) ---
@st.cache_resource
def load_my_model():
    model_path = 'model.keras'
    file_id = '1yYvHXYlkA2NRK4HGD5ANNDW5__mDP-C0'
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(model_path):
        with st.spinner("Initialisation des systèmes neuronaux..."):
            gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_my_model()

# --- 6. INTERFACE UTILISATEUR (Formulaire Patient) ---
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Informations Personnelles")
    nom = st.text_input("Nom")
    prenom = st.text_input("Prénom")
    date_naiss = st.date_input("Date de Naissance", min_value=datetime.date(1920, 1, 1))
    lieu_naiss = st.text_input("Lieu de Naissance")

with col2:
    st.subheader("📸 Analyse d'Imagerie")
    uploaded_file = st.file_uploader("Transférer l'IRM (JPG, PNG)", type=["jpg", "jpeg", "png"])

# --- 7. LOGIQUE DE PRÉDICTION & PDF ---
def create_pdf(nom, prenom, date_n, lieu_n, diag, conf, img_path):
    pdf = FPDF()
    pdf.add_page()
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "RAPPORT DE DIAGNOSTIC NEUROLOGIQUE (IA)", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, f"Généré par la plateforme de Houbad Douaa - {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align='C')
    
    # Infos Patient
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. INFORMATIONS PERSONNELLES", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Nom & Prénom : {nom.upper()} {prenom.capitalize()}", ln=True)
    pdf.cell(0, 8, f"Né(e) le : {date_n} à {lieu_n}", ln=True)
    
    # Image
    pdf.ln(5)
    pdf.image(img_path, x=60, w=90)
    
    # Résultats
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. RÉSULTATS DE L'ANALYSE PAR IA", ln=True)
    pdf.set_font("Arial", 'B', 14)
    color = (200, 0, 0) if "Pas de tumeur" not in diag else (0, 128, 0)
    pdf.set_text_color(*color)
    pdf.cell(0, 10, f"DIAGNOSTIC : {diag.upper()}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Indice de confiance du système : {conf:.2f}%", ln=True)
    
    return pdf.output(dest='S').encode('latin-1')

if uploaded_file and model:
    # Sauvegarde temporaire pour le PDF
    with open("temp_img.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="IRM Patient", width=400)
    
    if st.button("🧬 GÉNÉRER LE DIAGNOSTIC"):
        img_processed = img.resize((224, 224))
        img_array = np.array(img_processed) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = model.predict(img_array)
        class_names = ['Gliome', 'Méningiome', 'Pas de tumeur', 'Pituitaire']
        res_idx = np.argmax(preds)
        diag = class_names[res_idx]
        conf = np.max(preds) * 100
        
        st.markdown(f"""
            <div style="background-color: rgba(30, 58, 95, 0.1); padding: 20px; border-left: 5px solid #4A90E2; border-radius: 10px;">
                <h2 style="color: #1E3A5F;">Résultat de l'analyse : {diag}</h2>
                <h4>Confiance : {conf:.2f}%</h4>
            </div>
        """, unsafe_allow_html=True)
        
        if nom and prenom:
            pdf_out = create_pdf(nom, prenom, date_naiss, lieu_naiss, diag, conf, "temp_img.png")
            st.download_button("📥 Télécharger le Rapport Médical Officiel", pdf_out, f"Rapport_{nom}.pdf")

# --- 8. FOOTER & LINKEDIN ---
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center;">
        <p>Pour plus d'expertise biomédicale ou technique :</p>
        <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank">
            <button style="background-color: #0077B5; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-weight: bold;">
                for more information cliquer ici
            </button>
        </a>
    </div>
""", unsafe_allow_html=True)
