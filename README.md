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
   pip install -r requirements-project.txt
   ```

## Utilisation

1. **Lancer l'API FastAPI** :

   ```bash
   uvicorn app:app --reload
   ```

   L'API sera accessible à l'adresse `http://127.0.0.1:8000`.

2. **Lancer l'application Streamlit** :

   ```bash
   streamlit run app.py
   ```

   L'application sera accessible à l'adresse indiquée dans le terminal.

## Structure du projet

```
AI_Engineer_Projet_8/
├── app.py                   # Application Streamlit
├── app/                     # Dossier de l'API FastAPI
│   ├── __init__.py
│   ├── main.py              # Point d'entrée de l'API
│   └── ...
├── models/                  # Modèles de segmentation
│   └── best_model.keras
├── data/                    # Données d'entraînement et de test
│   ├── images/
│   └── masks/
├── requirements-project.txt # Dépendances complètes du projet
├── requirements.txt         # Dépendances pour Streamlit uniquement
└── README.md                # Présentation du projet
```
