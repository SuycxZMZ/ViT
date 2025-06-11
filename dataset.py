# dataset.py
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from config import config

def get_dataloaders():
    if not os.path.exists(config.dataset_path):
        os.makedirs(config.dataset_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(config.channels, -1, -1)),  # to 1x28x28
    ])

    dataset = datasets.MNIST(
        root=config.dataset_path,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=config.dataset_path,
        train=False,
        download=True,
        transform=transform
    )

    train_len = int(len(dataset) * 0.9)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    return train_loader, val_loader, test_loader