
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from supabase import create_client, Client
import requests  # <-- MODIFICARE 1: Importăm 'requests'
import io


# --- CONFIGUREAZĂ AICI PENTRU SUPABASE ---
SUPABASE_URL = "https://wsakatqrenlgljsaxfip.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndzYWthdHFyZW5sZ2xqc2F4ZmlwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI1NzQ2MzAsImV4cCI6MjA3ODE1MDYzMH0.GEE-uwc0DCcCxFmkLVwDlk2Q3OXSG_ImvkpK3NDY8Mo"  # Folosește cheia 'service_role'
# ------------------------------------

# --- MODIFICARE 2: Am eliminat configurarea httpx ---
# Inițializare simplă a clientului. Nu-l vom folosi pentru inserare,
# așa că bug-ul lui nu ne afectează.
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Client Supabase inițializat (dar nu va fi folosit pentru inserare).")
except Exception as e:
    print(f"Eroare FATALĂ la inițializarea clientului Supabase: {e}")
    exit()

app = Flask(__name__)
CORS(app)

# --- Încărcare Model (Rămâne neschimbat) ---
CLASS_NAMES = ['humans', 'robots']
MODEL_PATH = 'robot_human_classifier.pth'

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
except FileNotFoundError:
    print(f"Eroare FATALĂ: Fișierul model '{MODEL_PATH}' nu a fost găsit.")
    print("Asigură-te că ai rulat 'python train.py' mai întâi pentru a-l genera.")
    exit()
model.eval()
print(f"Modelul '{MODEL_PATH}' a fost încărcat cu succes.")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- Funcții Helper (Rămân neschimbate) ---

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return preprocess(image).unsqueeze(0)


def get_prediction(tensor):
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probabilities, 1)
        predicted_class = CLASS_NAMES[pred_idx.item()]
        confidence_score = confidence.item()
        return predicted_class, confidence_score


# --- MODIFICARE 3: Rescriem save_to_db() folosind 'requests' ---
def save_to_db(filename, predicted_class, confidence):
    """Salvează rezultatul în Supabase folosind API-ul REST, ocolind bug-ul httpx."""
    try:
        # Punctul final REST pentru tabela 'predictions'
        url = f"{SUPABASE_URL}/rest/v1/predictions"

        # Datele de inserat
        data_to_insert = {
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': confidence
        }

        # Antetele necesare pentru autentificare (API-ul REST)
        headers = {
            'apikey': SUPABASE_KEY,
            'Authorization': f'Bearer {SUPABASE_KEY}',  # Cheia service_role funcționează ca un token Bearer
            'Content-Type': 'application/json',
            'Prefer': 'return=minimal'  # Nu avem nevoie de răspunsul complet
        }

        # Facem cererea POST
        response = requests.post(url, headers=headers, json=data_to_insert)

        # Verificăm dacă a funcționat (codul 201 înseamnă "Created")
        if response.status_code != 201:
            # Dacă Supabase returnează o eroare, o afișăm în terminal
            print(f"Eroare la salvarea în Supabase (API): Status {response.status_code}, Răspuns: {response.text}")

    except Exception as e:
        # Eroare generală de rețea (ex: nu se poate conecta la Supabase)
        print(f"Eroare la salvarea în Supabase (Requests): {e}")


# --- ENDPOINTS (Rămân neschimbate) ---

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Niciun fișier încărcat'}), 400

    file = request.files['file']
    filename = file.filename

    if not filename:
        return jsonify({'error': 'Nume de fișier invalid'}), 400

    try:
        img_bytes = file.read()
        tensor = transform_image(img_bytes)

        predicted_class, confidence = get_prediction(tensor)

        # Apelăm noua funcție care folosește 'requests'
        save_to_db(filename, predicted_class, confidence)

        return jsonify({
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        print(f"Eroare la procesarea imaginii: {e}")
        return jsonify({'error': f"Eroare la procesarea imaginii: {str(e)}"}), 500


# --- Pornirea serverului (Rămâne neschimbată) ---
if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')