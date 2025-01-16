# Imports de la bibliothèque standard
import os
import logging
import requests

# Imports des bibliothèques tierces
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Imports des modules locaux
from api.custom_metrics import dice_bce_loss, dice_metric, iou_metric

# Configurer le logger
logger = logging.getLogger("image_segmentation_app")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Initialiser l'application FastAPI
app = FastAPI(title="Image Segmentation API", version="1.1")

# Chemin du modèle local et URL du modèle sur GitHub
model_path = "api/deployment_model/best_model.keras"
model_url = "https://github.com/Krock13/AI_Engineer_Projet_8_Traitez_les_images_pour_le_syst-me_embarque_d-une_voiture_autonome/blob/main/api/deployment_model/best_model.keras?raw=true"

def download_model(url, destination):
    """Télécharge le modèle depuis une URL."""
    try:
        logger.info(f"Téléchargement du modèle depuis {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Lève une erreur en cas de problème HTTP
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Modèle téléchargé avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement du modèle : {e}")
        raise FileNotFoundError("Impossible de télécharger le modèle.")

# Vérifier si le modèle existe localement, sinon le télécharger
if not os.path.exists(model_path):
    logger.warning("Le modèle n'a pas été trouvé localement. Téléchargement en cours...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    download_model(model_url, model_path)

model = load_model(model_path, custom_objects={
    'dice_bce_loss': dice_bce_loss,
    'dice_metric': dice_metric,
    'iou_metric': iou_metric
})

# Charger le modèle de segmentation
try:
    logger.info("Chargement du modèle de segmentation...")
    model = load_model(model_path, custom_objects={
        'dice_bce_loss': dice_bce_loss,
        'dice_metric': dice_metric,
        'iou_metric': iou_metric
    })
    logger.info("Modèle de segmentation chargé avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle : {e}")
    raise RuntimeError("Impossible de charger le modèle.")

def preprocess_image(image: UploadFile):
    """Prétraitement de l'image pour le modèle."""
    try:
        img = Image.open(image.file)
        img = img.resize((512, 256))  # Adapter la taille selon le modèle
        img_array = np.array(img) / 255.0  # Normalisation
        return np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement de l'image : {e}")
        raise HTTPException(status_code=400, detail="Erreur lors du prétraitement de l'image.")

@app.get("/")
def root():
    return {"message": "Bienvenue dans l'API de segmentation d'images !"}

@app.post("/predict")
def predict(image: UploadFile = File(...)):
    try:
        # Prétraitement de l'image
        input_data = preprocess_image(image)

        # Prédiction du masque
        predicted_mask = model.predict(input_data)
        predicted_mask = np.argmax(predicted_mask[0], axis=-1)  # Convertir en classes

        # Récupérer les couleurs de la palette Matplotlib "tab20"
        cmap = plt.get_cmap("tab20")
        color_map = {i: tuple((np.array(cmap(i)[:3]) * 255).astype(int)) for i in range(8)}

        # Créer une image colorée à partir du masque
        height, width = predicted_mask.shape
        mask_image = Image.new("RGB", (width, height))
        pixels = mask_image.load()

        for i in range(height):
            for j in range(width):
                pixels[j, i] = color_map[predicted_mask[i, j]]

        # Sauvegarder l'image du masque
        output_path = "mask_output_colored.png"
        mask_image.save(output_path)

        # Retourner l'image comme réponse
        return FileResponse(output_path, media_type="image/png", filename="mask_output_colored.png")

    except HTTPException as http_exc:
        # Si une exception HTTP a déjà été levée, renvoie-la directement
        logger.error(f"Erreur HTTP intentionnelle : {http_exc.detail}")
        raise http_exc

    except Exception as e:
        logger.error(f"Erreur interne lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction.")
