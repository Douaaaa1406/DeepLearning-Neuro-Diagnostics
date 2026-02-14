import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from fpdf import FPDF
import datetime
import os
import gdown

# --- 1. CONFIGURATION ET STYLE ---
st.set_page_config(page_title="NeuroScan AI | Houbad Douaa", page_icon="🧠", layout="wide")

# CSS pour le background MRI transparent et le style ingénieur
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
        background-color: rgba(255, 255, 255, 0.88); 
        z-index: -1;
    }
    .stHeader { color: #1E3A5F; }
</style>
""", unsafe_allow_html=True)

# --- 2. EN-TÊTE ET HORLOGE DYNAMIQUE ---
col_h1, col_h2 = st.columns([2, 1])

with col_h1:
    st.markdown(f"""
        <h1 style="color: #1E3A5F; margin-bottom: 0;">HOUBAD DOUAA</h1>
        <h3 style="color: #4A90E2; font-weight: 300; margin-top: 0;">Ingénierie Biomédicale & Data Science</h3>
    """, unsafe_allow_html=True)

with col_h2:
    # L'heure se met à jour à chaque interaction avec l'application
    now = datetime.datetime.now()
    st.markdown(f"""
        <div style="text-align: right; border: 2px solid #1E3A5F; padding: 10px; border-radius: 10px; background: rgba(255,255,255,0.7);">
            <b style="color: #1E3A5F;">📅 {now.strftime("%d/%m/%Y")}</b><br>
            <b style="color: #4A90E2;">⌚ {now.strftime("%H:%M:%S")}</b>
        </div>
    """, unsafe_allow_html=True)

# --- 3. CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_my_model():
    model_path = 'model.keras'
    file_id = '1yYvHXYlkA2NRK4HGD5ANNDW5__mDP-C0'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        with st.spinner("Initialisation du système expert..."):
            gdown.download(url, model_path, quiet=False)
    
    # Utilisation de compile=False pour ignorer les erreurs de structure d'entraînement
    return tf.keras.models.load_model(model_path, compile=False)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Erreur de chargement du modèle : {e}")
    model = None

# --- 4. FORMULAIRE PATIENT ---
st.markdown("---")
col_form1, col_form2 = st.columns(2)

with col_form1:
    st.subheader("📋 Identification Patient")
    nom = st.text_input("Nom")
    prenom = st.text_input("Prénom")
    date_n = st.date_input("Date de naissance", min_value=datetime.date(1920, 1, 1))
    lieu_n = st.text_input("Lieu de naissance")

with col_form2:
    st.subheader("🔬 Imagerie MRI")
    uploaded_file = st.file_uploader("Charger l'image DICOM/JPG/PNG", type=["jpg", "jpeg", "png"])

# --- 5. LOGIQUE PDF ---
def generate_pdf(nom, prenom, date_n, lieu_n, result, conf, img_path):
    pdf = FPDF()
    pdf.add_page()
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "RAPPORT D'ANALYSE BIOMÉDICALE - NEUROSCAN", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, f"Expert : Houbad Douaa | Date : {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align='C')
    
    # Infos
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. INFORMATIONS PERSONNELLES", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Patient : {nom.upper()} {prenom.capitalize()}", ln=True)
    pdf.cell(0, 8, f"Né(e) le : {date_n} à {lieu_n}", ln=True)
    
    # Image
    pdf.ln(5)
    pdf.image(img_path, x=60, w=90)
    
    # Diagnostic
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. RÉSULTATS DE L'ANALYSE", ln=True)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"DIAGNOSTIC : {result}", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Indice de confiance : {conf:.2f}%", ln=True)
    
    return pdf.output(dest='S').encode('latin-1')

# --- 6. ANALYSE ET RÉSULTATS ---
if uploaded_file and model:
    # Sauvegarde temporaire de l'image pour le PDF
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Aperçu de l'IRM", width=350)
    
    if st.button("🧬 GÉNÉRER LE DIAGNOSTIC"):
        # Prétraitement
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prédiction
        preds = model.predict(img_array)
        classes = ['Gliome', 'Méningiome', 'Pas de tumeur', 'Pituitaire']
        res_idx = np.argmax(preds)
        diag = classes[res_idx]
        conf = np.max(preds) * 100
        
        st.success(f"Analyse terminée : {diag} (Fiabilité : {conf:.2f}%)")
        
        # Bouton PDF
        if nom and prenom:
            pdf_data = generate_pdf(nom, prenom, date_n, lieu_n, diag, conf, "temp_image.png")
            st.download_button("📥 Télécharger le Rapport Médical Officiel", pdf_data, f"Rapport_{nom}.pdf")

# --- 7. FOOTER ---
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center;">
        <p>Expertise IA & Santé - Houbad Douaa</p>
        <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank">
            <button style="background-color: #0077B5; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-weight: bold;">
                for more information cliquer ici
            </button>
        </a>
    </div>
""", unsafe_allow_html=True)
