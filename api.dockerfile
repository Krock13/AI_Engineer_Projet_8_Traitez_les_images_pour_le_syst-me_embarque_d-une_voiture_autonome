# Dockerfile optimisé
FROM python:3.9-slim

# Répertoire de travail
WORKDIR /app

# Copier uniquement ce qui est nécessaire
COPY .dockerignore /app/.dockerignore
COPY main.py /app/
COPY custom_metrics.py /app/
COPY requirements-docker.txt /app/
COPY deployment_model/best_model.keras /app/deployment_model/

# Installer uniquement les dépendances nécessaires
RUN pip install --no-cache-dir -r requirements-docker.txt

# Exposer le port utilisé par FastAPI
EXPOSE 8000

# Commande par défaut
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]