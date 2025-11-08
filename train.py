import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset import get_dataloaders  # Importăm funcția de la Membrul 1
import time


# --- O funcție 'main' care conține toată logica ---
def main():
    # --- Setup ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Folosind dispozitivul: {device}")

    # Încărcare date
    DATA_DIR = 'dataset'
    BATCH_SIZE = 32
    NUM_EPOCHS = 10

    train_loader, val_loader, class_names = get_dataloaders(DATA_DIR, BATCH_SIZE)
    print(f"Clase găsite: {class_names}")

    # --- Model ---
    model = models.resnet18(weights='IMAGENET1K_V1')

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # --- Antrenare ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    print("Începe antrenarea...")

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
        # --- AICI ESTE MODIFICAREA ---
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
        # --- AICI ESTE A DOUA MODIFICARE ---
        val_epoch_acc = val_corrects.float() / len(val_loader.dataset)

        epoch_time = time.time() - start_time
        print(f"Epoca {epoch + 1}/{NUM_EPOCHS} ({epoch_time:.0f}s) - "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} - "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

    print("Antrenare finalizată.")

    # --- Salvarea Modelului ---
    MODEL_PATH = 'robot_human_classifier.pth'
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model salvat în {MODEL_PATH}")


# --- "Main guard-ul" pentru a proteja workerii pe macOS ---
if __name__ == '__main__':
    main()