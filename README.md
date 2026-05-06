# 🖼️ Traitement d'Image — Pipeline Vision Algorithmique (OpenCV)

Pipeline modulaire de traitement d'image en 8 étapes avec OpenCV : du prétraitement (filtrage, seuillage) à la segmentation et la détection de formes. Base algorithmique directement applicable à la vision embarquée (Raspberry Pi, caméra industrielle).

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green?logo=opencv)
![Vision](https://img.shields.io/badge/Vision-embarquée%20%7C%20algorithmique-orange)

---

## 🎯 Pourquoi ces techniques — le contexte embarqué

Ces algorithmes ne sont pas choisis arbitrairement. Ils constituent la **chaîne de traitement standard** avant tout système de détection basé sur la vision :

```
Image brute (caméra)
    │
    ▼
[1] Conversion niveaux de gris     → réduction dimensionnalité (3 canaux → 1)
    │
    ▼
[2] Filtrage Gaussien              → suppression bruit haute fréquence
    │
    ▼
[3] Filtrage Médian                → suppression bruit impulsionnel (sel-et-poivre)
    │
    ▼
[4] Seuillage adaptatif            → binarisation robuste aux variations d'éclairage
    │
    ▼
[5] Canny edge detection           → extraction contours (gradient Sobel + hystérésis)
    │
    ▼
[6] Érosion / Dilatation           → nettoyage morphologique, séparation objets
    │
    ▼
[7] Ouverture morphologique        → suppression faux positifs (petits bruits)
    │
    ▼
[8] Détection formes + bounding box → extraction des objets d'intérêt
```

En vision embarquée (Raspberry Pi + caméra, ligne de production, véhicule autonome), ce pipeline tourne **avant** tout modèle ML — c'est lui qui conditionne la qualité des features extraites.

---

## 🔬 Détail de chaque étape

### Filtrage (étapes 2–3)

```python
# Gaussien : convolution avec noyau gaussien — atténue le bruit aléatoire
cv2.GaussianBlur(img, ksize=(5,5), sigmaX=0)

# Médian : remplace chaque pixel par la médiane de son voisinage
# Supérieur au gaussien pour le bruit sel-et-poivre (pixels isolés)
cv2.medianBlur(img, ksize=5)
```

**Quand utiliser lequel ?** Gaussien pour le bruit de capteur CMOS, médian pour les défauts d'imagerie (hot pixels, compression JPEG).

### Seuillage adaptatif (étape 4)

```python
# Seuil local calculé dans une fenêtre 11×11 — résistant aux gradients d'éclairage
cv2.adaptiveThreshold(img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)
```

**Avantage vs seuil global :** un seuil global échoue si l'éclairage n'est pas uniforme (ombre partielle, rétro-éclairage). Le seuillage adaptatif s'adapte localement — essentiel en conditions réelles.

### Détection de contours Canny (étape 5)

```python
# Double seuil + hystérésis : seuil_bas=50 (contours faibles), seuil_haut=150 (forts)
cv2.Canny(img_floue, threshold1=50, threshold2=150)
```

L'algorithme de Canny (1986) reste la référence pour la détection de contours : calcul du gradient (Sobel), suppression des non-maxima, hystérésis pour relier les contours discontinus.

### Morphologie mathématique (étapes 6–7)

```python
kernel = np.ones((3,3), np.uint8)
erosion   = cv2.erode(img, kernel)         # Réduit les régions blanches
dilatation = cv2.dilate(img, kernel)       # Élargit les régions blanches
ouverture  = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # Érosion puis dilatation
```

**Ouverture** = supprimer les petits objets bruités sans déformer les grandes structures. Utilisé pour nettoyer le résultat du seuillage avant la détection.

### Détection de formes (étape 8)

```python
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if cv2.contourArea(cnt) > 500:           # Filtrer les petits artefacts
        x, y, w, h = cv2.boundingRect(cnt)  # Bounding box
        cv2.rectangle(img_couleur, (x,y), (x+w,y+h), (0,255,0), 2)
```

---

## 🖼️ Résultat du pipeline

Sur l'image de démonstration (formes géométriques + bruit) :

```
Image originale (600×400, 3 canaux, bruit sel-et-poivre)
    ↓
Niveaux de gris → Filtrage gaussien → Filtrage médian (bruit éliminé)
    ↓
Seuillage adaptatif → binarisation nette malgré les variations locales
    ↓
Canny → contours précis des formes géométriques
    ↓
Ouverture morphologique → suppression des artefacts résiduels
    ↓
Détection : 5 formes détectées (rectangle, cercle, ellipse, triangle, carré)
            avec bounding boxes annotées
```

> Voir `docs/screenshot_traitement.png` pour le résultat visuel complet.

---

## 🔗 Lien avec les systèmes embarqués

Ce pipeline est la base de la composante vision du projet **Voiture Robot Intelligente** (détection somnolence) — où OpenCV est utilisé pour le prétraitement des frames avant MediaPipe Face Mesh. Les mêmes techniques (filtrage, seuillage) sont applicables à :

- Détection de lignes de voie (TCRT5000 → caméra)
- Lecture de codes QR / RFID visuel
- Contrôle qualité industriel (caméra + convoyeur)
- Comptage de personnes / objets sur Raspberry Pi

---

## ⚠️ Limites connues

**Pipeline sans apprentissage.** Ces méthodes sont déterministes — elles ne s'adaptent pas à de nouvelles classes d'objets. Pour la détection d'objets génériques, un modèle CNN (YOLO, MobileNet) prendrait le relais après cette étape de prétraitement.

**Performance non optimisée pour temps réel.** Sur Raspberry Pi 4, ce pipeline tourne à ~15–20 fps sur des images 640×480. Pour des applications temps réel à 30 fps, il faudrait réduire la résolution ou utiliser OpenCV avec accélération NEON ARM.

**Seuils Canny empiriques.** Les seuils 50/150 sont adaptés à l'image de démonstration. Sur des images réelles, une détection automatique des seuils (méthode Otsu sur l'histogramme du gradient) est recommandée.

---

## 🗂️ Structure

```
traitement-image-opencv/
├── traitement_image.py   ← Pipeline 8 étapes (modulaire, chaque étape est une fonction)
├── docs/
│   └── screenshot_traitement.png
└── README.md
```

## ⚙️ Lancement

```bash
pip install opencv-python numpy matplotlib

# Avec image de démonstration (générée automatiquement)
python traitement_image.py

# Avec votre propre image
python traitement_image.py mon_image.jpg

# Sans fenêtre (génère un PNG du pipeline)
python traitement_image.py --no-display
```

## 🛠️ Technologies

**Python 3** · **OpenCV 4.5+** · **NumPy** · **Matplotlib**

## 👩‍💻 Auteure

**Vanelle Stéphanie MANGOUA** — Recherche d'alternance en IA & Systèmes Embarqués
