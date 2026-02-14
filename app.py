import streamlit as st
import tensorflow as tf
from PIL import Image, ImageEnhance
import numpy as np
from fpdf import FPDF
import datetime
import os
import gdown

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NeuroScan AI | Houbad Douaa", page_icon="🧠", layout="wide")

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
</style>
""", unsafe_allow_html=True)

# --- 2. EN-TÊTE ---
col_h1, col_h2 = st.columns([2, 1])
with col_h1:
    st.markdown('<p class="main-title">HOUBAD DOUAA</p>', unsafe_allow_html=True)
    st.write("Ingénierie Biomédicale & Data Science")

# --- 3. CHARGEMENT MODÈLE ---
@st.cache_resource
def load_my_model():
    model_path = 'brain_tumor_model_final.keras'
    file_id = '1yYvHXYlkA2NRK4HGD5ANNDW5__mDP-C0'
    if not os.path.exists(model_path):
        gdown.download(f'https://drive.google.com/uc?id={file_id}', model_path, quiet=False)
    
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.load_weights(model_path)
    return model

model = load_my_model()

# --- 4. FORMULAIRE ---
st.markdown("---")
col_p1, col_p2 = st.columns(2)
with col_p1:
    nom = st.text_input("Nom")
    prenom = st.text_input("Prénom")
    date_n = st.date_input("Date de naissance")
with col_p2:
    file = st.file_uploader("Charger l'IRM", type=["jpg", "png", "jpeg"])

# --- 5. ANALYSE AVEC REHAUSSEMENT ---
if file is not None and model is not None:
    img = Image.open(file).convert('RGB')
    
    if st.button("🧬 GÉNÉRER LE DIAGNOSTIC"):
        # ÉTAPE A : Rehaussement de contraste pour aider l'IA
        enhancer = ImageEnhance.Contrast(img)
        img_enhanced = enhancer.enhance(1.3) # Augmentation de 30% du contraste
        
        # ÉTAPE B : Prétraitement
        img_resized = img_enhanced.resize((224, 224), Image.LANCZOS)
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # ÉTAPE C : Prédiction
        prediction = model.predict(img_array)[0]
        classes = ['Gliome', 'Pas de tumeur', 'Méningiome', 'Pituitaire']
        res_idx = np.argmax(prediction)
        diag = classes[res_idx]
        conf = prediction[res_idx] * 100
        
        # Affichage
        st.write("### 📊 Analyse des probabilités :")
        cols = st.columns(4)
        for i in range(4):
            cols[i].metric(classes[i], f"{prediction[i]*100:.1f}%")

        # Alerte si conflit entre Pas de tumeur et Méningiome
        if (prediction[1] > 0.3 and prediction[2] > 0.3):
            st.warning("⚠️ **Zone d'incertitude détectée** : Le modèle hésite entre l'absence de tumeur et un méningiome périphérique. Une double lecture par un expert est fortement recommandée.")

        st.success(f"Résultat : {diag} ({conf:.2f}%)")

        # --- 6. RAPPORT PDF ---
        if nom and prenom:
            img.save("temp.jpg", "JPEG")
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Times", 'B', 18) # Style Cambria-like
            pdf.cell(0, 20, "RAPPORT MEDICAL NEUROSCAN AI", ln=True, align='C')
            
            pdf.ln(10)
            pdf.set_font("Times", 'B', 12)
            pdf.cell(0, 10, f"PATIENT : {nom.upper()} {prenom.capitalize()}", ln=True)
            pdf.image("temp.jpg", x=60, w=90)
            
            pdf.set_y(pdf.get_y() + 95)
            pdf.set_font("Times", 'B', 14)
            pdf.cell(0, 15, f"DIAGNOSTIC : {diag.upper()}", 1, ln=True, align='C')
            
            # Footer demandé
            pdf.set_y(-40)
            pdf.set_font("Times", 'B', 10)
            pdf.cell(0, 10, f"Modèle : NeuroScan-V1 | Ingénieur : HOUBAD DOUAA", ln=True, align='C')
            pdf.set_font("Times", 'I', 9)
            pdf.multi_cell(0, 5, "AVERTISSEMENT : Travail basé sur l'IA. Veuillez consulter votre médecin.", align='C')
            
            st.download_button("📥 Télécharger le PDF", pdf.output(dest='S').encode('latin-1'), f"Rapport_{nom}.pdf")
