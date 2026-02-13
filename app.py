import streamlit as st

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="NeuroScan AI | Houbad Douaa", page_icon="🧠")

# --- STYLE CSS POUR LE BACKGROUND ---
# On utilise une image MRI avec une opacité réduite (0.05 à 0.1) 
# pour qu'elle reste discrète en arrière-plan.
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
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(255, 255, 255, 0.85); /* Calque blanc semi-transparent */
    z-index: -1;
}}

.stApp {{
    background: transparent;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- SIGNATURE HOUBAD DOUAA ---
st.markdown(f"""
    <div style="text-align: center; padding: 20px;">
        <h3 style="color: #4A90E2; font-family: 'Helvetica'; font-weight: 300; margin-bottom: 0;">PLATEFORME DE DIAGNOSTIC</h3>
        <h1 style="color: #1E3A5F; font-size: 3.5em; font-weight: 900; margin-top: 0; letter-spacing: 2px;">HOUBAD DOUAA</h1>
        <p style="color: #555; font-style: italic;">Intelligence Artificielle appliquée à la Neuro-Radiologie</p>
        <hr style="border: 0; height: 2px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(74, 144, 226, 0.75), rgba(0, 0, 0, 0));">
    </div>
    """, unsafe_allow_html=True)
