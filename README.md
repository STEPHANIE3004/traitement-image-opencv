# 🖼️ Traitement d'Image — Pipeline OpenCV

Pipeline complet de traitement d'image en Python avec OpenCV, incluant filtrage, morphologie, détection de contours et segmentation. Projet académique réalisé dans le cadre d'un cursus en Intelligence Artificielle & Systèmes Embarqués.

## 📌 Fonctionnalités

| Technique | Description |
|-----------|-------------|
| Seuillage adaptatif | Binarisation locale (Gaussian + Mean) |
| Filtrage Gaussien | Lissage par convolution gaussienne |
| Filtrage Médian | Réduction du bruit impulsionnel |
| Détection Canny | Extraction des contours par gradient |
| Morphologie | Érosion, dilatation, ouverture, fermeture |
| Détection de formes | Contours + bounding boxes annotés |
| Pipeline complet | Enchaînement automatique de toutes les étapes |

## 🗂️ Structure du projet

```
traitement-image-opencv/
├── traitement_image.py   # Script principal
├── README.md
└── images/               # (optionnel) images de test
```

## ⚙️ Installation

```bash
pip install opencv-python numpy matplotlib
```

## 🚀 Utilisation

### Avec une image existante
```bash
python traitement_image.py --image chemin/vers/image.jpg
```

### Avec l'image de démonstration (générée automatiquement)
```bash
python traitement_image.py --demo
```

### Afficher toutes les étapes du pipeline
```bash
python traitement_image.py --demo --pipeline
```

## 📊 Résultats

Le script génère une grille de visualisation 2×4 avec :
- Image originale en niveaux de gris
- Seuillage adaptatif Gaussian
- Seuillage adaptatif Mean
- Filtrage Gaussien (σ=1)
- Filtrage Médian (k=5)
- Détection de contours Canny
- Opérations morphologiques (érosion / dilatation)
- Détection et annotation des formes

## 🧠 Concepts couverts

- Traitement d'image bas-niveau (filtrage spatial)
- Morphologie mathématique (éléments structurants)
- Détection de contours (opérateur de Canny)
- Segmentation par seuillage adaptatif
- Visualisation et annotation avec OpenCV & Matplotlib

## 🛠️ Technologies

- **Python 3.x**
- **OpenCV** (`cv2`) — traitement d'image
- **NumPy** — calculs matriciels
- **Matplotlib** — visualisation

## 👩‍💻 Auteure

**Vanelle Stéphanie MANGOUA DJOUSSEU**  
Étudiante en IA & Systèmes Embarqués — Recherche d'alternance  
[LinkedIn](https://linkedin.com/in/vanelle-mangoua) · [GitHub](https://github.com/vanellemangoua)
