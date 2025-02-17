# Traitez les images pour le système embarqué d'une voiture autonome

## À propos

Ce projet vise à développer une application de segmentation d'images pour les systèmes embarqués de voitures autonomes. L'objectif est de traiter et d'analyser des images afin d'identifier et de segmenter différents éléments de l'environnement routier, contribuant ainsi à la navigation sécurisée du véhicule.

## Table des matières

- [À propos](#à-propos)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)

## Prérequis

Avant de commencer, assurez-vous d'avoir les éléments suivants installés sur votre machine :

- **Python 3.8 ou supérieur** : [Télécharger Python](https://www.python.org/downloads/)
- **pip** : Gestionnaire de paquets Python
- **Git** : [Télécharger Git](https://git-scm.com/downloads)

## Installation

1. **Cloner le dépôt** :

   ```bash
   git clone https://github.com/Krock13/AI_Engineer_Projet_8_Traitez_les_images_pour_le_syst-me_embarque_d-une_voiture_autonome.git
   cd AI_Engineer_Projet_8_Traitez_les_images_pour_le_syst-me_embarque_d-une_voiture_autonome
   ```

2. **Créer un environnement virtuel** :

   ```bash
   python -m venv env
   source env/bin/activate  # Sur Windows : env\Scripts\activate
   ```

3. **Installer les dépendances** :

   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

1. **Lancer l'API FastAPI** :

   ```bash
   uvicorn api.main:app --reload
   ```

   L'API sera accessible à l'adresse `http://127.0.0.1:8000`.

2. **Lancer l'application Streamlit** :

   ```bash
   streamlit run web/app.py
   ```

   L'application sera accessible à l'adresse indiquée dans le terminal.

## Structure du projet

```
AI_Engineer_Projet_8/
├── api/                     # Dossier pour l'API FastAPI
│   ├── deployment_model/    # Modèle de segmentation
│   ├── custom_metrics.py    # Fonctions de métriques
│   ├── main.py              # Point d'entrée de l'API
│   ├── requirements.txt     # Dépendances de l'API
├── web/                     # Dossier pour l'application Streamlit
│   ├── app.py               # Point d'entrée de l'application Streamlit
│   ├── assets/              # Contient les images et masques utilisés
│   ├── requirements.txt     # Dépendances de Streamlit
├── data/                    # Données d'entraînement et de test
│   ├── test/                # Données de test
│   ├── train/               # Données d'entraînement
│   └── val/                 # Données de validation
├── models/                  # Modèles de segmentation
├── requirements.txt # Dépendances complètes pour l'ensemble du projet
├── requirements-docker.txt  # Dépendances pour le déploiement Docker
└── README.md                # Présentation globale du projet
```
