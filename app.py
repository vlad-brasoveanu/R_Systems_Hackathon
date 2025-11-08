import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from supabase import create_client, Client, ClientOptions  # <-- MODIFICARE 1: Importăm ClientOptions
import httpx  # <-- MODIFICARE 2: Importăm httpx
import io
import json

# --- CONFIGUREAZĂ AICI PENTRU SUPABASE ---
SUPABASE_URL = "https://[URL-UL-PROIECTULUI-TAU].supabase.co"
SUPABASE_KEY = "[CHEIA-TA-SERVICE-ROLE]"
# ------------------------------------

# --- MODIFICARE 3: SOLUȚIA PENTRU BUG-UL IPv6 ---
# Creăm un client httpx personalizat care preferă adresele IPv4
# "local_address="127.0.0.1"" forțează legarea la interfața IPv4 locală
custom_httpx_client = httpx.Client(transport=httpx.HTTPTransport(local_address="127.0.0.1"))
# Creăm opțiunile clientului Supabase
supabase_options = ClientOptions(httpx_client=custom_httpx_client)
# ------------------------------------


# --- Inițializare Conexiuni ---
try:
    # Injectăm opțiunile personalizate în clientul Supabase
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=supabase_options)
    print("Conectat la Supabase cu succes (folosind client IPv4 personalizat).")
except Exception as e:
    print(f"Eroare FATALĂ la conectarea cu Supabase: {e}")
    print("Verifică URL-ul și Cheia API în app.py")
    exit()

app = Flask(__name__)
# Permitem cereri de la frontend-ul web (necesar pentru dezvoltare locală)
CORS(app)

# --- Încărcare Model ---
# Asigură-te că numele claselor sunt în ordinea corectă (alfabetică, cum le încarcă ImageFolder)
CLASS_NAMES = ['humans', 'robots']
MODEL_PATH = 'robot_human_classifier.pth'

# 1. Re-creăm arhitectura modelului (ResNet-18)
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

# 2. Încărcăm parametrii antrenați
try:
    # Încărcăm modelul pe CPU, deoarece Flask este pentru inferență, nu antrenare
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
except FileNotFoundError:
    print(f"Eroare FATALĂ: Fișierul model '{MODEL_PATH}' nu a fost găsit.")
    print("Asigură-te că ai rulat 'python train.py' mai întâi pentru a-l genera.")
    exit()

model.eval()  # Foarte important: setează modelul în modul de evaluare!
print(f"Modelul '{MODEL_PATH}' a fost încărcat cu succes.")

# 3. Defineste transformările pentru o singură imagine (la fel ca la validare)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda img: img.convert('RGB')),  # Asigurăm 3 canale
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- Funcții Helper ---

def transform_image(image_bytes):
    """Transformă octeții imaginii într-un tensor Pytorch."""
    image = Image.open(io.BytesIO(image_bytes))
    return preprocess(image).unsqueeze(0)  # Adaugă o dimensiune pentru batch


def get_prediction(tensor):
    """Rulează predicția."""
    with torch.no_grad():
        outputs = model(tensor)
        # Aplicăm Softmax pentru a obține probabilități
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Extragem clasa de top și confidența
        confidence, pred_idx = torch.max(probabilities, 1)

        predicted_class = CLASS_NAMES[pred_idx.item()]
        confidence_score = confidence.item()

        return predicted_class, confidence_score


def save_to_db(filename, predicted_class, confidence):
    """Salvează rezultatul în tabela 'predictions' din SupABASE."""
    try:
        data_to_insert = {
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': confidence
        }
        # Inserăm în tabela 'predictions' (pe care ai creat-o în Supabase)
        response = supabase.table('predictions').insert(data_to_insert).execute()

    except Exception as e:
        print(f"Eroare la salvarea în Supabase: {e}")
        # Nu oprim aplicația, doar logăm eroarea și mergem mai departe


# --- ENDPOINTS (RUTELE SERVERULUI) ---

@app.route('/')
def home():
    """Servește pagina web principală (index.html)"""
    # Trimite fișierul 'index.html' din folderul curent ('.')
    return send_from_directory('.', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint-ul API care primește imaginea și returnează predicția."""
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

        # Salvează rezultatul în Supabase
        save_to_db(filename, predicted_class, confidence)

        # Returnează răspunsul JSON către frontend
        return jsonify({
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        # Gestionează și erorile de procesare (ex: fișier corupt)
        print(f"Eroare la procesarea imaginii: {e}")
        return jsonify({'error': f"Eroare la procesarea imaginii: {str(e)}"}), 500


# --- Pornirea serverului ---
if __name__ == '__main__':
    # MODIFICARE: Am revenit la '0.0.0.0' pentru a fi accesibil de 'ngrok'
    # Am păstrat portul 5001 pentru a evita conflictul cu AirPlay
    # Problema IPv6 ar trebui rezolvată acum prin instalarea 'requests'
    app.run(debug=True, port=5001, host='0.0.0.0')