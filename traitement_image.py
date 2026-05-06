"""
Traitement d'Image — Vision Algorithmique
Pipeline modulaire de traitement d'image avec OpenCV
Auteur : Vanelle Stéphanie MANGOUA DJOUSSEU — ESIEA
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# ─── Fonctions de traitement ──────────────────────────────────────────────────

def charger_image(chemin):
    """Charge une image en niveaux de gris et en couleur."""
    img_couleur = cv2.imread(chemin)
    if img_couleur is None:
        raise FileNotFoundError(f"Image introuvable : {chemin}")
    img_gris = cv2.cvtColor(img_couleur, cv2.COLOR_BGR2GRAY)
    return img_couleur, img_gris

def seuillage_adaptatif(img_gris):
    """Seuillage adaptatif gaussien — résistant aux variations d'éclairage."""
    return cv2.adaptiveThreshold(
        img_gris, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

def filtrage_gaussien(img_gris, ksize=5):
    """Lissage gaussien pour réduction du bruit."""
    return cv2.GaussianBlur(img_gris, (ksize, ksize), 0)

def filtrage_median(img_gris, ksize=5):
    """Filtre médian — efficace contre le bruit sel-et-poivre."""
    return cv2.medianBlur(img_gris, ksize)

def detection_contours_canny(img_gris, seuil_bas=50, seuil_haut=150):
    """Détection de contours par l'algorithme de Canny."""
    img_floue = filtrage_gaussien(img_gris, ksize=5)
    return cv2.Canny(img_floue, seuil_bas, seuil_haut)

def erosion(img_binaire, iterations=1):
    """Érosion morphologique."""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(img_binaire, kernel, iterations=iterations)

def dilatation(img_binaire, iterations=1):
    """Dilatation morphologique."""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(img_binaire, kernel, iterations=iterations)

def ouverture(img_binaire):
    """Ouverture = érosion puis dilatation — supprime les petits bruits."""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(img_binaire, cv2.MORPH_OPEN, kernel)

def fermeture(img_binaire):
    """Fermeture = dilatation puis érosion — comble les petits trous."""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(img_binaire, cv2.MORPH_CLOSE, kernel)

def detection_contours_formes(img_originale, img_binaire):
    """Détecte et dessine les contours des formes sur l'image originale."""
    contours, _ = cv2.findContours(
        img_binaire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    img_resultat = img_originale.copy()
    # Filtrer les petits contours (bruit)
    contours_filtres = [c for c in contours if cv2.contourArea(c) > 100]
    cv2.drawContours(img_resultat, contours_filtres, -1, (0, 255, 0), 2)
    print(f"  → {len(contours_filtres)} formes détectées")
    return img_resultat

def pipeline_complet(chemin_image, afficher=True, sauvegarder=True):
    """
    Pipeline complet de traitement d'image.
    Applique dans l'ordre : chargement → gris → filtrage → seuillage →
    détection contours → morphologie → visualisation.
    """
    print(f"\n[Pipeline] Traitement de : {chemin_image}")

    # 1. Chargement
    img_couleur, img_gris = charger_image(chemin_image)
    print(f"  → Image chargée : {img_gris.shape[1]}×{img_gris.shape[0]} px")

    # 2. Filtrage
    img_gaussienne = filtrage_gaussien(img_gris)
    img_mediane    = filtrage_median(img_gris)

    # 3. Seuillage
    img_seuil = seuillage_adaptatif(img_gris)

    # 4. Détection de contours
    img_canny = detection_contours_canny(img_gris)

    # 5. Morphologie
    img_ouverte  = ouverture(img_seuil)
    img_fermee   = fermeture(img_seuil)

    # 6. Détection des formes
    img_formes = detection_contours_formes(img_couleur, img_ouverte)

    # 7. Visualisation
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Pipeline de Traitement d\'Image — Vision Algorithmique', fontsize=14, fontweight='bold')

    donnees = [
        (img_couleur[:,:,::-1], 'Image Originale (RGB)',    None),
        (img_gris,              'Niveaux de Gris',           'gray'),
        (img_gaussienne,        'Filtrage Gaussien (k=5)',   'gray'),
        (img_mediane,           'Filtrage Médian (k=5)',     'gray'),
        (img_seuil,             'Seuillage Adaptatif',       'gray'),
        (img_canny,             'Contours Canny',            'gray'),
        (img_ouverte,           'Ouverture Morphologique',   'gray'),
        (img_formes[:,:,::-1],  'Formes Détectées',          None),
    ]

    for ax, (img, titre, cmap) in zip(axes.flat, donnees):
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(titre, fontsize=9)
        ax.axis('off')

    plt.tight_layout()

    if sauvegarder:
        nom_sortie = os.path.splitext(chemin_image)[0] + '_pipeline.png'
        plt.savefig(nom_sortie, dpi=150, bbox_inches='tight')
        print(f"  → Résultat sauvegardé : {nom_sortie}")

    if afficher:
        plt.show()

    return {
        'original':  img_couleur,
        'gris':      img_gris,
        'gaussien':  img_gaussienne,
        'median':    img_mediane,
        'seuil':     img_seuil,
        'canny':     img_canny,
        'ouverture': img_ouverte,
        'fermeture': img_fermee,
        'formes':    img_formes,
    }

def creer_image_demo():
    """Crée une image de démonstration avec des formes géométriques."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 240
    # Formes géométriques
    cv2.rectangle(img, (50, 50), (180, 150), (30, 80, 200), -1)
    cv2.circle(img, (300, 100), 70, (200, 50, 30), -1)
    cv2.ellipse(img, (480, 100), (90, 50), 30, 0, 360, (50, 180, 50), -1)
    cv2.fillPoly(img, [np.array([[80,200],[200,350],[20,320]])], (180, 100, 200))
    cv2.rectangle(img, (250, 200), (420, 370), (50, 200, 200), -1)
    # Bruit sel-et-poivre
    nb_pixels = 500
    coords = [np.random.randint(0, s, nb_pixels) for s in img.shape[:2]]
    img[coords[0], coords[1]] = [255, 255, 255]
    coords2 = [np.random.randint(0, s, nb_pixels) for s in img.shape[:2]]
    img[coords2[0], coords2[1]] = [0, 0, 0]
    demo_path = 'image_demo.png'
    cv2.imwrite(demo_path, img)
    print(f"Image de démonstration créée : {demo_path}")
    return demo_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline de traitement d\'image avec OpenCV')
    parser.add_argument('image', nargs='?', help='Chemin vers l\'image à traiter')
    parser.add_argument('--no-display', action='store_true', help='Ne pas afficher la fenêtre')
    args = parser.parse_args()

    if args.image:
        pipeline_complet(args.image, afficher=not args.no_display)
    else:
        print("Aucune image fournie — création d'une image de démonstration...")
        demo = creer_image_demo()
        pipeline_complet(demo, afficher=not args.no_display)
