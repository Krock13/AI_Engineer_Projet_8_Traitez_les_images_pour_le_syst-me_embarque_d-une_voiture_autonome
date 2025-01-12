from fastapi.testclient import TestClient
from api.main import app  # Importer l'application FastAPI
from io import BytesIO

# Créer un client de test
client = TestClient(app)

def test_root_endpoint():
    """
    Teste si l'endpoint racine (/) retourne le message attendu.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue dans l'API de segmentation d'images !"}


def test_predict_endpoint_with_image():
    """
    Teste l'endpoint /predict avec une image valide.
    """
    # Charger une petite image de test depuis un fichier ou en créer une
    image_path = "tests/test_image.png"
    with open(image_path, "rb") as image_file:
        files = {"image": ("test_image.png", BytesIO(image_file.read()), "image/png")}

    # Envoyer une requête POST à l'API
    response = client.post("/predict", files=files)

    # Vérifier la réponse
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

def test_predict_endpoint_without_image():
    """
    Teste l'endpoint /predict sans envoyer d'image.
    """
    response = client.post("/predict")
    assert response.status_code == 422
    assert "detail" in response.json()
    assert response.json()["detail"][0]["msg"] == "Field required"

def test_predict_endpoint_with_invalid_image():
    """
    Teste l'endpoint /predict avec un fichier non valide.
    """
    files = {"image": ("test.txt", BytesIO(b"This is not an image"), "text/plain")}
    response = client.post("/predict", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Erreur lors du prétraitement de l'image."
