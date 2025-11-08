import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_dataloaders(data_dir, batch_size=32):
    """Creează DataLoaders pentru antrenare și validare."""

    # Redimensionare, Normalizare și Augmentare (opțional)
    # Imaginile trebuie să fie de 224x224 pentru ResNet
    # Valorile de normalizare sunt standard pentru modelele pre-antrenate pe ImageNet
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),  # Augmentare
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Încarcă datele folosind ImageFolder
    # Presupunem că ai o singură structură mare și o împărțim
    # Alternativ, poți folosi direct folderele 'train' și 'val'

    # Metoda 1: Dacă ai foldere 'train' și 'val' separate
    train_dataset = datasets.ImageFolder(data_dir + '/train', data_transforms['train'])
    val_dataset = datasets.ImageFolder(data_dir + '/val', data_transforms['val'])

    # Metoda 2: Dacă ai un singur folder 'dataset' și vrei să-l împarți
    # full_dataset = datasets.ImageFolder(data_dir, data_transforms['train'])
    # train_size = int(0.8 * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    # train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # Aplică transformările de validare setului de validare (necesită un pic mai mult cod)
    # Pentru sprint, Metoda 1 (foldere separate) e mai rapidă.

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    class_names = train_dataset.classes

    return train_loader, val_loader, class_names