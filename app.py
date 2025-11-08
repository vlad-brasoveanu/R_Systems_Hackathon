import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
import sqlite3
import io
import json
from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- Importă CORS
from supabase import create_client, Client
import io


# --- Setup Aplicație ---
app = Flask(__name__)
CORS(app)

# --- Setup Model ---
# Trebuie să RE-CREĂM arhitectura modelului înainte de a încărca parametrii
# Asigură-te că folosești aceleași clase ca în train.py
CLASS_NAMES = ['humans', 'robots']  # ATENȚIE: Ordinea trebuie să fie corectă!
MODEL_PATH = 'robot_human_classifier.pth'

# 1. Inițiază modelul (la fel ca în train.py)
model = models.resnet18()  # Nu avem nevoie de 'weights' aici
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

# 2. Încarcă parametrii salvați
# Folosim map_location pentru a ne asigura că rulează pe CPU dacă e necesar
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()  # Foarte important: setează modelul în modul de evaluare!

# 3. Definește transformările pentru o singură imagine (la fel ca 'val' în train.py)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- Funcții Helper ---

def transform_image(image_bytes):
    """Transformă octeții imaginii într-un tensor Pytorch."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return preprocess(image).unsqueeze(0)  # Adaugă o dimensiune pentru batch (batch_size=1)


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
    """Salvează rezultatul în baza de date SQLite."""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (filename, predicted_class, confidence) VALUES (?, ?, ?)",
        (filename, predicted_class, confidence)
    )
    conn.commit()
    conn.close()


# --- Endpoint API ---

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

        # Salvare în baza de date
        save_to_db(filename, predicted_class, confidence)

        # Returnează răspunsul
        return jsonify({
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)