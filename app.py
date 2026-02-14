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

# --- 2. STYLE CSS (Background MRI & Transparence) ---
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
    .main-title { color: #1E3A5F; font-size: 3em; font-weight: 900; margin-bottom: 0; }
    .sub-title { color: #4A90E2; font-size: 1.4em; font-weight: 300; margin-top: 0; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #1E3A5F; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 3. EN-TÊTE AVEC HORLOGE DYNAMIQUE ---
col_h1, col_h2 = st.columns([2, 1])

with col_h1:
    st.markdown('<p class="main-title">HOUBAD DOUAA</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Ingénierie Biomédicale & Data Science</p>', unsafe_allow_html=True)

with col_h2:
    # Horloge temps réel
    now = datetime.datetime.now()
    st.markdown(f"""
        <div style="text-align: right; border: 2px solid #1E3A5F; padding: 10px; border-radius: 10px; background-color: rgba(255,255,255,0.6);">
            <span style="color: #1E3A5F; font-size: 1.2em; font-weight: bold;">📅 {now.strftime("%d/%m/%Y")}</span><br>
            <span style="color: #4A90E2; font-size: 1.1em; font-weight: bold;">⌚ {now.strftime("%H:%M:%S")}</span>
        </div>
    """, unsafe_allow_html=True)

# --- 4. CHARGEMENT DU MODÈLE (CORRECTION TECHNIQUE) ---
@st.cache_resource
def load_my_model():
    model_path = 'model.keras'
    file_id = '1yYvHXYlkA2NRK4HGD5ANNDW5__mDP-C0'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        with st.spinner("Chargement sécurisé du modèle..."):
            gdown.download(url, model_path, quiet=False)
    
    try:
        # Tentative avec le chargeur universel qui ignore les erreurs de structure Keras
        # On utilise TFSMLayer pour charger le modèle comme un graphe TensorFlow pur
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        # Si ça échoue encore, on utilise le format SavedModel interne
        st.warning("Adaptation du format de données en cours...")
        return tf.saved_model.load(model_path)
# --- 5. FORMULAIRE PATIENT ---
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Identification Patient")
    nom = st.text_input("Nom")
    prenom = st.text_input("Prénom")
    date_n = st.date_input("Date de naissance", min_value=datetime.date(1920, 1, 1))
    lieu_n = st.text_input("Lieu de naissance")

with col2:
    st.subheader("🔬 Acquisition d'Image")
    file = st.file_uploader("Transférer l'IRM cérébrale (JPG/PNG)", type=["jpg", "jpeg", "png"])

# --- 6. FONCTION GÉNÉRATION PDF PROFESSIONNEL ---
def generate_medical_pdf(nom, prenom, date_n, lieu_n, result, confidence, img_path):
    pdf = FPDF()
    pdf.add_page()
    
    # En-tête
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(30, 58, 95)
    pdf.cell(0, 10, "RAPPORT DE DIAGNOSTIC NEUROLOGIQUE (IA)", ln=True, align='C')
    
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Expertise : Houbad Douaa | Date de l'analyse : {datetime.datetime.now().strftime('%d/%m/%Y à %H:%M')}", ln=True, align='C')
    
    # Section Patient
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. INFORMATIONS PERSONNELLES DU PATIENT", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Identité : {nom.upper()} {prenom.capitalize()}", ln=True)
    pdf.cell(0, 8, f"Date de naissance : {date_n} (Lieu : {lieu_n})", ln=True)
    
    # Image IRM
    pdf.ln(5)
    pdf.image(img_path, x=60, w=90)
    pdf.ln(5)
    
    # Résultats
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. CONCLUSIONS DE L'ANALYSE PAR DEEP LEARNING", ln=True)
    pdf.set_font("Arial", 'B', 14)
    if "Pas de tumeur" in result:
        pdf.set_text_color(34, 139, 34)
    else:
        pdf.set_text_color(200, 0, 0)
    pdf.cell(0, 10, f"RÉSULTAT : {result.upper()}", ln=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Indice de fiabilité du système : {confidence:.2f}%", ln=True)
    
    return pdf.output(dest='S').encode('latin-1')

# --- 7. TRAITEMENT ET PRÉDICTION ---

if file is not None and model is not None:
    # Sauvegarde temporaire
    with open("temp_mri.png", "wb") as f:
        f.write(file.getbuffer())
        
    img = Image.open(file).convert('RGB')
    st.image(img, caption="IRM Patient chargée avec succès", width=350)
    
    if st.button("🧬 LANCER L'ANALYSE BIOMÉDICALE"):
        # Prétraitement (Dimension standard 224x224)
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prédiction
        prediction = model.predict(img_array)
        classes = ['Gliome', 'Méningiome', 'Pas de tumeur', 'Pituitaire']
        res_idx = np.argmax(prediction)
        diag_final = classes[res_idx]
        conf_final = np.max(prediction) * 100
        
        # Affichage
        st.markdown(f"""
            <div style="background-color: white; border-left: 10px solid #1E3A5F; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 15px rgba(0,0,0,0.1);">
                <h2 style="color: #1E3A5F; margin: 0;">Diagnostic : {diag_final}</h2>
                <h4 style="color: #4A90E2; margin: 0;">Confiance statistique : {conf_final:.2f}%</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Bouton PDF
        if nom and prenom:
            pdf_bytes = generate_medical_pdf(nom, prenom, date_n, lieu_n, diag_final, conf_final, "temp_mri.png")
            st.download_button(
                label="📥 Télécharger le Rapport Médical Officiel",
                data=pdf_bytes,
                file_name=f"NeuroScan_Report_{nom}.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("Veuillez remplir les informations du patient pour générer le rapport.")

# --- 8. FOOTER & LINKEDIN ---
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; padding: 20px;">
        <p style="color: #555;">Expertise IA & Santé par Houbad Douaa</p>
        <a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank" style="text-decoration: none;">
            <button style="background-color: #0077B5; color: white; border: none; padding: 12px 25px; border-radius: 30px; cursor: pointer; font-size: 16px; font-weight: bold;">
                for more information cliquer ici
            </button>
        </a>
    </div>
""", unsafe_allow_html=True)
