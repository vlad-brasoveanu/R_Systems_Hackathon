import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset import get_dataloaders
import time
import json  # <-- Import nou
import matplotlib.pyplot as plt  # <-- Import nou
import numpy as np  # <-- Import nou


def main():
    """Funcția principală care rulează antrenarea."""

    # --- Setup Dispozitiv ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Folosind dispozitivul: {device}")

    # --- Configurare ---
    DATA_DIR = 'dataset'
    BATCH_SIZE = 32
    NUM_EPOCHS = 20  # MODIFICAT: Mărim numărul de epoci pentru performanță

    # --- Încărcare Date ---
    try:
        train_loader, val_loader, class_names = get_dataloaders(DATA_DIR, BATCH_SIZE)
    except FileNotFoundError:
        print(f"Eroare: Folderul '{DATA_DIR}' nu a fost găsit.")
        print("Asigură-te că ai creat structura 'dataset/train/...' și 'dataset/val/...'")
        return

    # VERIFICARE: Acum class_names ar trebui să fie ['humans', 'others', 'robots']
    print(f"Clase găsite: {class_names}")

    # --- Definire Model (Transfer Learning) ---
    model = models.resnet18(weights='IMAGENET1K_V1')

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    # MODIFICAT: Acum modelul va avea 3 ieșiri (pentru cele 3 clase)
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)

    # --- Definire Antrenare ---
    criterion = nn.CrossEntropyLoss()
    # MODIFICAT: Trecem la optimizatorul Adam pentru o antrenare mai rapidă și mai bună
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    print("Începe antrenarea...")

    # --- Colectare statistici pentru raport ---
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # --- Bucla de Antrenare ---
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # Faza de antrenare
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.float() / len(train_loader.dataset)

        # Faza de validare
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.float() / len(val_loader.dataset)

        epoch_time = time.time() - start_time
        print(f"Epoca {epoch + 1}/{NUM_EPOCHS} ({epoch_time:.0f}s) - "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} - "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # Salvează statisticile în istoric
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())

    print("Antrenare finalizată.")

    # --- Salvarea Modelului ---
    MODEL_PATH = 'robot_human_classifier.pth'
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model salvat în {MODEL_PATH}")

    # --- Salvarea Raportului de Antrenare (JSON) ---
    REPORT_PATH_JSON = 'training_report.json'
    with open(REPORT_PATH_JSON, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Raport de antrenare salvat în {REPORT_PATH_JSON}")

    # --- Generarea Graficului de Antrenare (PNG) ---
    REPORT_PATH_PLOT = 'training_plot.png'
    epochs_range = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(12, 5))

    # Grafic Acuratețe
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Acuratețe Antrenare')
    plt.plot(epochs_range, history['val_acc'], label='Acuratețe Validare')
    plt.title('Acuratețe Antrenare și Validare')
    plt.xlabel('Epoca')
    plt.ylabel('Acuratețe')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Grafic Pierdere
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Pierdere Antrenare')
    plt.plot(epochs_range, history['val_loss'], label='Pierdere Validare')
    plt.title('Pierdere Antrenare și Validare')
    plt.xlabel('Epoca')
    plt.ylabel('Pierdere')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()  # Ajustează layout-ul pentru a preveni suprapunerea
    plt.savefig(REPORT_PATH_PLOT)
    print(f"Graficul de antrenare salvat în {REPORT_PATH_PLOT}")


# --- "Main guard-ul" pentru a proteja workerii pe macOS ---
if __name__ == '__main__':
    main()