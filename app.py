import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os

# Configuration des chemins
IMAGE_DIR = "assets/images/"
MASK_DIR = "assets/masks/"
API_URL = "https://autonomouscarapp-gpeqc3c6e4ajdeca.francecentral-01.azurewebsites.net/predict"

# Récupérer la liste des images disponibles
def get_image_list(image_dir):
    return sorted([f for f in os.listdir(image_dir) if f.endswith('_leftImg8bit.png')])

# Mapper l'image avec le masque correspondant
def get_corresponding_mask(image_name):
    return image_name.replace("_leftImg8bit.png", "_gtFine_color.png")

# Fonction pour appeler l'API
def get_predicted_mask(image_path):
    try:
        with open(image_path, "rb") as image_file:
            files = {"image": image_file}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
            else:
                st.error(f"Erreur API : {response.status_code} - {response.text}")
                return None
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")
        return None

# Interface utilisateur Streamlit
st.title("Application de Segmentation d'Images")
st.write("Choisis une image dans la liste pour voir le masque attendu et la segmentation prédite.")

# Charger la liste des images
image_list = get_image_list(IMAGE_DIR)

# Vérifier s'il y a des images disponibles
if not image_list:
    st.error("Aucune image disponible dans le dossier.")
else:
    # Interface pour choisir une image
    selected_image = st.selectbox("Sélectionne une image :", image_list)

    # Afficher l'image sélectionnée
    if selected_image:
        st.image(os.path.join(IMAGE_DIR, selected_image), caption="Image Originale", use_container_width=True)

        # Chemin du masque attendu
        corresponding_mask = get_corresponding_mask(selected_image)
        expected_mask_path = os.path.join(MASK_DIR, corresponding_mask)

        # Vérifier si le masque attendu existe
        if os.path.exists(expected_mask_path):
            expected_mask = Image.open(expected_mask_path)
        else:
            st.warning(f"Masque attendu non trouvé pour {selected_image}.")

        # Bouton pour lancer la prédiction
        if st.button("Lancer la prédiction"):
            predicted_mask = get_predicted_mask(os.path.join(IMAGE_DIR, selected_image))

            if predicted_mask:
                # Afficher les résultats
                st.image(expected_mask, caption="Masque Attendu", use_container_width=True)
                st.image(predicted_mask, caption="Masque Prédit", use_container_width=True)
