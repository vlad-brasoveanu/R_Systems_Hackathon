import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests  # Singura noastră dependență de rețea!
import io
import json
import base64
import numpy as np
import cv2
import uuid

# --- Importuri pentru Grad-CAM ---
from torchcam import methods as cam_methods
# NU mai folosim overlay_mask, vom face fuzionarea manual2

# --- CONFIGUREAZIA AICI PENTRU SUPABASE ---
SUPABASE_URL = "https://wsakatqrenlgljsaxfip.supabase.co"  # <-- COMPLETEAZĂ AICI
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndzYWthdHFyZW5sZ2xqc2F4ZmlwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI1NzQ2MzAsImV4cCI6MjA3ODE1MDYzMH0.GEE-uwc0DCcCxFmkLVwDlk2Q3OXSG_ImvkpK3NDY8Mo"  # <-- COMPLETEAZĂ AICI
BUCKET_NAME = "R_Systems_Hackathon"  # Numele bucket-ului (TREBUIE SĂ FIE PUBLIC)
# ------------------------------------

# --- Inițializare API-uri ---
app = Flask(__name__)
CORS(app)

# --- Inițializare Client Bază de Date (Metoda REST/Requests) ---
supabase_db_headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}
supabase_table_url = f"{SUPABASE_URL}/rest/v1/predictions"
print("Client Supabase (pentru Baza de Date) inițializat.")

# --- Inițializare Client Storage (Metoda REST/Requests) ---
supabase_storage_url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}"
supabase_storage_headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}"
}
print("Client Supabase (pentru Storage) inițializat.")

# --- Încărcare Model ---
CLASS_NAMES = ['humans', 'robots']
MODEL_PATH = 'robot_human_classifier.pth'
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print(f"Modelul '{MODEL_PATH}' a fost încărcat cu succes.")
except FileNotFoundError:
    print(f"Eroare FATALĂ: Fișierul model '{MODEL_PATH}' nu a fost găsit.")
    print("Asigură-te că ai rulat 'python train.py' mai întâi pentru a-l genera.")
    exit()

# --- Inițializare Extractor Grad-CAM ---
target_layer = model.layer4[1].conv2
# Activăm gradienții pe stratul țintă pentru a permite Grad-CAM să funcționeze
for param in target_layer.parameters():
    param.requires_grad = True
print(f"[DEBUG Grad-CAM] Activare requires_grad pe stratul țintă.")

cam_extractor = cam_methods.GradCAM(model, target_layer=target_layer)
print("Extractorul Grad-CAM a fost inițializat.")

# --- Definire Transformări ---
# Transformarea de input pentru model (dimensiune fixă)
preprocess_model = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionare forțată
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Transformare simplă pentru heatmap (doar conversie în tensor)
preprocess_heatmap = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor()
])


# --- Funcții Helper (Actualizate) ---

def save_to_db(filename, predicted_class, confidence, image_url):
    """Salvează în Supabase folosind API-ul REST (metoda requests)."""
    try:
        payload = {
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'image_url': image_url
        }
        print(f"[DEBUG DB] Se încearcă salvarea în DB: {filename}")
        response = requests.post(supabase_table_url, headers=supabase_db_headers, json=payload, timeout=5)

        if response.status_code == 201:
            print("[DEBUG DB] Salvare în DB reușită.")
        else:
            if "Failed to parse" not in response.text:
                print(f"Eroare Supabase DB (non-201): {response.text}")

    except Exception as e:
        if "Failed to parse" not in str(e):
            print(f"Eroare la salvarea în Supabase DB (Requests): {str(e)}")


def upload_to_storage(pil_image):
    """Încarcă imaginea PIL în Supabase Storage folosind Requests."""
    try:
        storage_filename = f"img_{uuid.uuid4()}.jpg"
        print(f"[DEBUG Storage] Început încărcare în Storage (Bucket: {BUCKET_NAME})...")

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()
        print(f"[DEBUG Storage] Imagine convertită în bytes (dimensiune: {len(image_bytes)})")

        upload_url = f"{supabase_storage_url}/{storage_filename}"
        storage_headers = {**supabase_storage_headers, "Content-Type": "image/jpeg"}

        response = requests.post(upload_url, data=image_bytes, headers=storage_headers, timeout=10)

        if response.status_code != 200:
            print(f"Eroare la încărcarea în Supabase Storage (non-200): {response.text}")
            return None

        public_url_string = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{storage_filename}"
        print(f"[DEBUG Storage] URL public obținut: {public_url_string}")

        return public_url_string

    except Exception as e:
        print(f"Eroare la încărcarea în Supabase Storage: {str(e)}")
        return None


def process_image_and_predict(img_bytes, filename="image"):
    """
    Funcție helper centralizată care procesează o imagine, generează Grad-CAM,
    o încarcă în Storage și salvează totul în DB.
    """
    print("[DEBUG Procesare] Început procesare imagine...")
    try:
        pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        # Obținem dimensiunea originală
        original_width, original_height = pil_image.size
        print(f"[DEBUG Procesare] Dimensiune originală: {original_width}x{original_height}")

        # Pregătim tensorul pentru model (redimensionat la 224x224)
        tensor_model = preprocess_model(pil_image).unsqueeze(0)
        print("[DEBUG Procesare] Imagine convertită în tensor (224x224).")

    except Exception as e:
        print(f"Eroare la procesarea imaginii: {e}")
        return {"error": "Fișierul pare a fi corupt sau nu este o imagine."}

    # --- Încărcare în Storage ---
    public_image_url = upload_to_storage(pil_image)

    # --- Predicție și Grad-CAM ---
    heatmap_base64 = None
    try:
        print("[DEBUG Grad-CAM] Intrat în blocul 'enable_grad'.")
        with torch.enable_grad():
            tensor_model.requires_grad = True
            outputs = model(tensor_model)
            print("[DEBUG Grad-CAM] Predicție model finalizată.")

            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)

            predicted_class = CLASS_NAMES[pred_idx.item()]
            confidence_score = confidence.item()
            class_idx = pred_idx.item()
            print(f"[DEBUG Grad-CAM] Clasa prezisă: {predicted_class} ({confidence_score:.2f})")

            # --- Generare Hartă Termică ---
            print("[DEBUG Grad-CAM] Se extrage harta de activare...")
            activation_map = cam_extractor(class_idx, outputs)[0]
            activation_map_np = activation_map.squeeze(0).cpu().numpy()
            print(f"[DEBUG Grad-CAM] Formă hartă (numpy): {activation_map_np.shape}")

            # Normalizare 0-1
            map_min, map_max = activation_map_np.min(), activation_map_np.max()
            if map_max > map_min:
                activation_map_np = (activation_map_np - map_min) / (map_max - map_min)
            print("[DEBUG Grad-CAM] Hartă normalizată (0-1).")

            # Conversie în imagine 0-255 (uint8)
            activation_map_uint8 = (activation_map_np * 255).astype(np.uint8)

            # --- MODIFICARE: Redimensionare la DIMENSIUNEA ORIGINALĂ ---
            heatmap_resized = cv2.resize(activation_map_uint8, (original_width, original_height))
            print(f"[DEBUG Grad-CAM] Hartă redimensionată la {original_width}x{original_height}.")

            # --- MODIFICARE: Aplică paleta de culori HOT ---
            heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_HOT)
            print("[DEBUG Grad-CAM] Hartă colorată (COLORMAP_HOT).")

            # Conversie imagine originală PIL în OpenCV (Numpy array)
            original_image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Fuzionarea imaginii originale cu harta termică
            alpha = 0.5  # Transparența hărții
            blended_image_cv = cv2.addWeighted(heatmap_colored, alpha, original_image_cv, 1 - alpha, 0)
            print("[DEBUG Grad-CAM] Fuzionare finalizată.")

            # Conversie finală în Base64
            _G, buffer = cv2.imencode('.jpg', blended_image_cv)
            b64_string = base64.b64encode(buffer).decode('utf-8')
            heatmap_base64 = f"data:image/jpeg;base64,{b64_string}"
            print("[DEBUG Grad-CAM] Fuzionare convertită în Base64.")

    except Exception as e:
        print(f"[DEBUG Grad-CAM] EROARE la generarea Grad-CAM: {e}")

    # --- Salvare în DB ---
    save_to_db(filename, predicted_class, confidence_score, public_image_url)

    # --- Returnare Răspuns ---
    print("[DEBUG Procesare] Se returnează JSON către frontend.")
    return {
        'filename': filename,
        'predicted_class': predicted_class,
        'confidence': confidence_score,
        'heatmap_image': heatmap_base64
    }


# --- Endpoint-uri API (Backend) ---

@app.route('/')
def home():
    """Servește pagina web principală (index.html)"""
    return send_from_directory('.', 'index.html')


@app.route('/predict', methods=['POST'])
def predict_from_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Niciun fișier încărcat'}), 400
    file = request.files['file']
    img_bytes = file.read()
    response_data = process_image_and_predict(img_bytes, file.filename)
    return jsonify(response_data)


@app.route('/predict_url', methods=['POST'])
def predict_from_url():
    data = request.json
    if 'url' not in data:
        return jsonify({'error': 'URL lipsă'}), 400
    url = data['url']
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        img_bytes = response.content
        filename = url.split('/')[-1]
        response_data = process_image_and_predict(img_bytes, filename)
        return jsonify(response_data)
    except Exception as e:
        print(f"Eroare la procesarea URL-ului: {e}")
        return jsonify({'error': 'Eroare la procesarea URL-ului.'}), 500


@app.route('/get_recent', methods=['GET'])
def get_recent_predictions():
    """Endpoint pentru a returna ultimele 5 predicții din istoric."""
    try:
        params = {"order": "timestamp.desc", "limit": 5, "select": "*,image_url"}
        response = requests.get(supabase_table_url, headers=supabase_db_headers, params=params, timeout=5)
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        if "Failed to parse" not in str(e):
            print(f"Eroare la preluarea istoricului: {e}")
        return jsonify({'error': 'Nu s-a putut încărca istoricul.'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')