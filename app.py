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

# --- 2. EN-TÊTE ---
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

# --- 3. CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_my_model():
    model_path = 'brain_tumor_model_final.keras'
    file_id = '1yYvHXYlkA2NRK4HGD5ANNDW5__mDP-C0'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        with st.spinner("Chargement du cerveau artificiel..."):
            gdown.download(url, model_path, quiet=False)
    
    try:
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
    except Exception as e:
        st.error(f"Erreur technique : {e}")
        return None

model = load_my_model()

# --- 4. FORMULAIRE ---
st.markdown("---")
col_p1, col_p2 = st.columns(2)
with col_p1:
    st.subheader("📋 Identification Patient")
    nom = st.text_input("Nom")
    prenom = st.text_input("Prénom")
    date_n = st.date_input("Date de naissance", value=datetime.date(2000, 1, 1))
    lieu_n = st.text_input("Lieu de naissance")

with col_p2:
    st.subheader("🔬 Image IRM")
    file = st.file_uploader("Charger le scan (JPG, PNG)", type=["jpg", "jpeg", "png"])

# --- 5. LOGIQUE DE DIAGNOSTIC ---
if file is not None and model is not None:
    img = Image.open(file).convert('RGB')
    st.image(img, width=300, caption="Scan IRM chargé")
    
    if st.button("🧬 GÉNÉRER LE DIAGNOSTIC"):
        # 1. Prétraitement rigoureux
        img_resized = img.resize((224, 224), Image.LANCZOS)
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 2. Prédiction brute
        prediction = model.predict(img_array)[0]
        
        # 3. ORDRE DES CLASSES RÉALIGNÉ (Index 1 = Pas de tumeur)
        classes = ['Gliome', 'Pas de tumeur', 'Méningiome', 'Pituitaire']
        
        res_idx = np.argmax(prediction)
        diag = classes[res_idx]
        conf = prediction[res_idx] * 100
        
        # 4. Affichage des probabilités
        st.write("### 📊 Analyse des probabilités :")
        cols = st.columns(4)
        for i in range(4):
            cols[i].metric(classes[i], f"{prediction[i]*100:.1f}%")

        st.markdown(f"""
            <div style="background-color: white; border-left: 10px solid #1E3A5F; padding: 20px; border-radius: 10px; margin-top:20px;">
                <h2 style="color: #1E3A5F; margin:0;">Résultat : {diag}</h2>
                <h4 style="color: #4A90E2; margin:0;">Confiance : {conf:.2f}%</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # --- 6. GÉNÉRATION DU RAPPORT PDF ---
        if nom and prenom:
            img.save("temp_report.jpg", "JPEG")
            pdf = FPDF()
            pdf.add_page()
            
            # Utilisation de Times (équivalent Serif de Cambria dans FPDF standard)
            font_style = "Times" 
            
            # Header
            pdf.set_fill_color(30, 58, 95)
            pdf.rect(0, 0, 210, 40, 'F')
            pdf.set_text_color(255, 255, 255)
            pdf.set_font(font_style, 'B', 18)
            pdf.cell(0, 20, "RAPPORT MEDICAL NEUROSCAN AI", ln=True, align='C')
            
            pdf.ln(25)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font(font_style, 'B', 12)
            pdf.cell(0, 10, " 1. INFORMATIONS DU PATIENT", 1, ln=True)
            pdf.set_font(font_style, '', 11)
            pdf.cell(95, 10, f" Nom : {nom.upper()}", 1)
            pdf.cell(95, 10, f" Prenom : {prenom.capitalize()}", 1, ln=True)
            
            pdf.ln(10)
            pdf.image("temp_report.jpg", x=60, w=90)
            pdf.set_y(pdf.get_y() + 95)
            
            pdf.set_font(font_style, 'B', 12)
            pdf.cell(0, 10, " 2. RESULTATS DU MODELE", 1, ln=True)
            pdf.set_font(font_style, 'B', 14)
            pdf.cell(0, 15, f" DIAGNOSTIC : {diag.upper()} ({conf:.2f}%)", 1, ln=True, align='C')
            
            # Pied de page (Footer)
            pdf.set_y(-40)
            pdf.set_font(font_style, 'B', 10)
            pdf.cell(0, 10, f"Modele : NeuroScan-V1 | Ingenieur : HOUBAD DOUAA", ln=True, align='C')
            pdf.set_font(font_style, 'I', 9)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(0, 5, "AVERTISSEMENT : Travail base sur l'IA. Veuillez consulter votre medecin.", align='C')
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("📥 Télécharger le Rapport PDF", pdf_bytes, f"Rapport_{nom}.pdf")

# --- 7. FOOTER LINKEDIN ---
st.markdown("---")
st.markdown('<div style="text-align: center;"><a href="https://www.linkedin.com/in/douaa-houbad-006b6a305" target="_blank"><button style="background-color: #0077B5; color: white; border: none; padding: 12px 25px; border-radius: 30px; cursor: pointer; font-weight: bold;">for more information cliquer ici</button></a></div>', unsafe_allow_html=True)
