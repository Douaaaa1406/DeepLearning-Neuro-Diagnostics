import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from fpdf import FPDF
import datetime
import os
import gdown

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="NeuroScan AI | Houbad Douaa", page_icon="🧠", layout="wide")

# Style CSS (MRI Background & Professional UI)
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
    .main-title { color: #1E3A5F; font-size: 3em; font-weight: 900; margin-bottom: 0; }
    .sub-title { color: #4A90E2; font-size: 1.4em; font-weight: 300; margin-top: 0; }
</style>
""", unsafe_allow_html=True)

# --- 2. EN-TÊTE AVEC HORLOGE ---
col_h1, col_h2 = st.columns([2, 1])
with col_h1:
    st.markdown('<p class="main-title">HOUBAD DOUAA</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Ingénierie Biomédicale & Data Science</p>', unsafe_allow_html=True)

with col_h2:
    now = datetime.datetime.now()
    st.markdown(f"""
        <div style="text-align: right; border: 2px solid #1E3A5F; padding: 10px; border-radius: 10px; background-color: rgba(255,255,255,0.6);">
            <span style="color: #1E3A5F; font-size: 1.1em; font-weight: bold;">📅 {now.strftime("%d/%m/%Y")}</span><br>
            <span style="color: #4A90E2; font-size: 1em; font-weight: bold;">⌚ {now.strftime("%H:%M:%S")}</span>
        </div>
    """, unsafe_allow_html=True)

# --- 3. CHARGEMENT DU MODÈLE (ARCHITECTURE MOBILE-NET V2) ---
@st.cache_resource
def load_my_model():
    model_path = 'model.keras'
    file_id = '1yYvHXYlkA2NRK4HGD5ANNDW5__mDP-C0'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)
    
    try:
        # Reconstruction de l'architecture exacte identifiée précédemment
        base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        model.load_weights(model_path)
        return model
    except Exception as e:
        st.error(f"Erreur technique de chargement : {e}")
        return None

model = load_my_model()

# --- 4. FORMULAIRE PATIENT ---
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 Identification Patient")
    nom = st.text_input("Nom")
    prenom = st.text_input("Prénom")
    date_n = st.date_input("Date de naissance", value=datetime.date(2000, 1, 1))
    lieu_n = st.text_input("Lieu de naissance")

with col2:
    st.subheader("🔬 Imagerie Médicale")
    file = st.file_uploader("Charger l'IRM (JPG, PNG)", type=["jpg", "jpeg", "png"])

# --- 5. FONCTION PDF STRUCTURÉE ---
def generate_professional_pdf(nom, prenom, date_n, lieu_n, result, confidence, img_path):
    pdf = FPDF()
    pdf.add_page()
    
    # En-tête bleu
    pdf.set_fill_color(30, 58, 95)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 20, "RAPPORT D'ANALYSE NEUROSCAN AI", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 5, "Expertise Neurologique assistée par Deep Learning", ln=True, align='C')
    
    # Corps du rapport
    pdf.ln(25)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, " 1. INFORMATIONS DU PATIENT", 1, ln=True, fill=True)
    
    pdf.set_font("Arial", '', 11)
    pdf.cell(95, 10, f" Nom : {nom.upper()}", 1)
    pdf.cell(95, 10, f" Prénom : {prenom.capitalize()}", 1, ln=True)
    pdf.cell(95, 10, f" Date de naissance : {date_n}", 1)
    pdf.cell(95, 10, f" Lieu : {lieu_n}", 1, ln=True)
    
    # Image au centre
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, " 2. CLICHÉ IRM EXAMINÉ", 0, ln=True)
    pdf.image(img_path, x=60, w=90)
    
    # Espace pour l'image
    pdf.set_y(pdf.get_y() + 95)
    
    # Résultats
    pdf.set_fill_color(235, 245, 255)
    pdf.cell(0, 10, " 3. RÉSULTATS DU MODÈLE NEUROSCAN", 1, ln=True, fill=True)
    
    color = (34, 139, 34) if "Pas de tumeur" in result else (200, 0, 0)
    pdf.set_text_color(*color)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 15, f" DIAGNOSTIC : {result.upper()}", 1, ln=True, align='C')
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'I', 11)
    pdf.cell(0, 10, f" Indice de certitude : {confidence:.2f}%", 0, ln=True, align='C')
    
    # --- PIED DE PAGE (Footer demandé) ---
    pdf.set_y(-45)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, f"Modèle : NeuroScan-MobileNetV2 (Version 1.0) | Ingénieur : HOUBAD DOUAA", ln=True, align='C')
    pdf.set_font("Arial", 'BI', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, "AVERTISSEMENT : Ce travail est basé sur l'IA. Il s'agit d'un outil d'aide au diagnostic. Veuillez consulter votre médecin ou un neuro-radiologue pour une interprétation clinique officielle.", align='C')
    
    return pdf.output(dest='S').encode('latin-1')

# --- 6. LOGIQUE D'ANALYSE ---


if file is not None and model is not None:
    img = Image.open(file).convert('RGB')
    st.image(img, width=300, caption="Aperçu du cliché IRM")
    
    if st.button("🧬 GÉNÉRER LE DIAGNOSTIC & RAPPORT"):
        # Conversion JPEG pour le PDF
        temp_path = "temp_print.jpg"
        img.save(temp_path, "JPEG")
        
        # Prétraitement
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prédiction
        prediction = model.predict(img_array)
        classes = ['Gliome', 'Méningiome', 'Pas de tumeur', 'Pituitaire']
        res_idx = np.argmax(prediction)
        diag = classes[res_idx]
        conf = np.max(prediction) * 100
        
        st.success(f"Analyse terminée avec succès : {diag}")
        
        if nom and prenom:
            pdf_data = generate_professional_pdf(nom, prenom, date_n, lieu_n, diag, conf, temp_path)
            st.download_button("📥 Télécharger le Rapport Officiel (PDF)", pdf_data, f"Rapport_NeuroScan_{nom}.pdf")
        else:
            st.warning("Veuillez remplir les informations patient pour activer le téléchargement.")

# --- 7. FOOTER LINKEDIN ---
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center;">
        <p style="color: #555;">Développement et Algorithmes par Houbad Douaa</p>
        <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank">
            <button style="background-color: #0077B5; color: white; border: none; padding: 12px 25px; border-radius: 30px; cursor: pointer; font-weight: bold;">
                for more information cliquer ici
            </button>
        </a>
    </div>
""", unsafe_allow_html=True)
