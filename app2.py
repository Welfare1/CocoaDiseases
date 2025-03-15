import os
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import joblib  # Importation de joblib

# Désactiver les optimisations oneDNN dans TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# --------------------------
# 1. Chargement des modèles
# --------------------------

st.sidebar.title("Configuration")

try:
    # Charger le modèle YOLOv8 personnalisé pour la détection des fèves de cacao.
    model_yolo = YOLO("Model/model_cacao.pt")

    # Charger le modèle de classification basé sur ConvNeXtSmall
    # Charger le modèle de classification basé sur ConvNeXtSmall (déjà entraîné)
    classifier = tf.keras.models.load_model("Model/modele_convnext_small_mine_tf(e50).keras", compile=False)

    svm_clf = joblib.load("Model/best_svm_SIEL.pkl")

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
    "class2": (255, 255, 0)  # Jaune
}

# --------------------------
# 2. Fonction de traitement
# --------------------------

def extract_features(image, model):
    img_resized = cv2.resize(image, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_preprocessed = tf.keras.applications.convnext.preprocess_input(np.array(img_rgb, dtype=np.float32))
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
    features = model.predict(img_preprocessed)
    return features.squeeze()


def process_image(image, use_svm=False):
    if image is None:
        st.error("L'image chargée est invalide. Veuillez réessayer.")
        return None
    
    # Appliquer le modèle YOLOv8 pour détecter les fèves
    results = model_yolo(image)

    # Récupérer les boîtes englobantes
    boxes = results[0].boxes
    if boxes is None:
        st.warning("Aucune fève détectée dans l'image.")
        return image

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
        return image
    
    # Copie de l'image pour annoter les résultats
    output_img = image.copy()

    # Pour chaque détection, extraire la ROI, la classifier et afficher le résultat
    for (x1, y1, x2, y2, conf_det) in detections_list:
        # Extraire la région d'intérêt correspondant à la fève
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # Préparation de la ROI pour le modèle de classification
        features = extract_features(roi, classifier)
        
        if use_svm:
            pred_class = svm_clf.predict([features])[0]
            confidence_class = 1.0  # SVM ne fournit pas directement une probabilité
        else:
            pred_prob = classifier.predict(np.array([features]))  # Correction de la ligne
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

    return output_img

# --------------------------
# 3. Interface Streamlit
# --------------------------

st.title("Détection et Classification des Fèves de Cacao 🍫")
st.write("Téléchargez une image pour détecter et classifier les fèves de cacao.")

# Option pour choisir le modèle de classification
model_choice = st.sidebar.radio("Choisissez le modèle de classification :", ["ConvNeXt seul", "ConvNeXt + SVM"])

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lire l'image téléchargée
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if image is not None:
        # Traiter l'image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        use_svm = (model_choice == "ConvNeXt + SVM")
        output_img = process_image(image, use_svm=use_svm)
        
        if output_img is not None:
            # Afficher les images
            st.image(image, caption='Image Originale', use_column_width=True)
            st.image(output_img, caption='Image avec Détections et Classifications', use_column_width=True)
    else:
        st.error("Impossible de lire l'image. Veuillez réessayer avec une autre image.")
