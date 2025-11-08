import os
from PIL import Image

# Definește folderul principal al setului de date
dataset_root = 'dataset'
subfolders = ['train/humans', 'train/robots', 'val/humans', 'val/robots', 'val/others', 'val/others']

corrupted_count = 0

print("--- Începe verificarea imaginilor corupte ---")

for subfolder in subfolders:
    folder_path = os.path.join(dataset_root, subfolder)

    if not os.path.exists(folder_path):
        print(f"Atenție: Folderul {folder_path} nu există. Se sare peste.")
        continue

    print(f"\nVerific în: {folder_path}...")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        try:
            # Încearcă să deschizi imaginea și să îi încarci datele
            with Image.open(file_path) as img:
                img.load()  # Forțează citirea datelor imaginii

        except (IOError, OSError, Image.UnidentifiedImageError) as e:
            # Dacă deschiderea sau încărcarea eșuează, fișierul e corupt
            print(f"  [ȘTERS] Fișier corupt detectat: {filename} (Eroare: {e})")

            # Șterge fișierul
            try:
                os.remove(file_path)
                corrupted_count += 1
            except Exception as remove_e:
                print(f"    Eroare la ștergerea fișierului: {remove_e}")

print("\n--- Verificare finalizată ---")
print(f"Total fișiere corupte găsite și șterse: {corrupted_count}")
