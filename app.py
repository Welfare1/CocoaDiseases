import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image

# --------------------------
# 1. Chargement des modèles
# --------------------------

st.sidebar.title("Configuration")

try:
    # Charger le modèle YOLOv8 personnalisé pour la détection des fèves de cacao.
    model_yolo = YOLO("Model/model_cacao.pt")

    # Charger le modèle de classification basé sur ConvNeXtSmall
    classifier = tf.keras.models.load_model("Model/model_convnextsmall_mine0.keras", compile=False)

    st.sidebar.success("Modèles chargés avec succès ✅")
except Exception as e:
    st.sidebar.error(f"Erreur lors du chargement des modèles : {e}")
    st.stop()

# Définir les noms des classes de maladies (3 classes dans cet exemple)
target_names = ["class0", "class1", "class2"]

# Couleurs des cadres pour chaque classe
class_colors = {
    "class0": (255, 0, 0),   # Rouge
    "class1": (0, 0, 255),   # Bleu
    "class2": (255, 255,0)  # Jaune
}

# --------------------------
# 2. Fonction de traitement
# --------------------------

def process_image(image):
    # Vérifier que l'image n'est pas vide
    if image is None:
        st.error("L'image chargée est invalide. Veuillez réessayer.")
        return None, None

    # Redimensionner l'image pour l'affichage
    display_image = cv2.resize(image, (600, 400))
    
    # Appliquer le modèle YOLOv8 pour détecter les fèves
    results = model_yolo(image)

    # Récupérer les boîtes englobantes
    boxes = results[0].boxes
    if boxes is None:
        st.warning("Aucune fève détectée dans l'image.")
        return image, display_image

    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()

    # Filtrer les détections avec un seuil de confiance
    detections_list = []
    confidence_threshold = st.sidebar.slider("Seuil de confiance YOLO", 0.0, 1.0, 0.5, 0.05)
    
    for i in range(len(xyxy)):
        if confidences[i] < confidence_threshold:
            continue
        x1, y1, x2, y2 = map(int, xyxy[i])
        detections_list.append((x1, y1, x2, y2, confidences[i]))

    if not detections_list:
        st.warning("Aucune fève n'a été retenue après filtrage.")
        return image, display_image

    # Copie de l'image pour annoter les résultats
    output_img = image.copy()

    # Pour chaque détection, extraire la ROI, la classifier et afficher le résultat
    for (x1, y1, x2, y2, conf_det) in detections_list:
        # Extraire la région d'intérêt correspondant à la fève
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # Préparation de la ROI pour le modèle de classification
        roi_resized = cv2.resize(roi, (224, 224))
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        roi_preprocessed = tf.keras.applications.convnext.preprocess_input(np.array(roi_rgb, dtype=np.float32))
        roi_preprocessed = np.expand_dims(roi_preprocessed, axis=0)

        # Prédiction de la classe de maladie pour la ROI
        pred_prob = classifier.predict(roi_preprocessed)
        pred_class = np.argmax(pred_prob, axis=1)[0]
        confidence_class = np.max(pred_prob)

        # Création du label à afficher
        class_name = target_names[pred_class]
        label_text = f"{class_name} ({confidence_class:.2f})"

        # Dessiner la bounding box sur l'image de sortie
        color = class_colors.get(class_name, (255, 255, 255))  # Blanc par défaut
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 3)  # Épaisseur du cadre augmentée à 3
        cv2.putText(output_img, label_text, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return output_img, display_image

# --------------------------
# 3. Interface Streamlit
# --------------------------

st.title("Détection et Classification des Fèves de Cacao 🍫")
st.write("Téléchargez une image pour détecter et classifier les fèves de cacao.")

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lire l'image téléchargée
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if image is not None:
        # Traiter l'image
        output_img, display_image = process_image(image)

        if output_img is not None:
            # Afficher les images
            st.image(display_image, caption='Image Originale Redimensionnée', use_column_width=True)
            st.image(output_img, caption='Image avec Détections et Classifications', use_column_width=True)
    else:
        st.error("Impossible de lire l'image. Veuillez réessayer avec une autre image.")
