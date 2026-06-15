import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config

def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((config.MNIST_MEAN,), (config.MNIST_STD,))
    ])

    train_set = datasets.MNIST(config.DATA_DIR, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(config.DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, test_loader