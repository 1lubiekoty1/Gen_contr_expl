import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config

def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean & std
    ])

    train_set = datasets.MNIST(config.DATA_DIR, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(config.DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, test_loader