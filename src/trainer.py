import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_dataloaders
from src.utils import get_device , save_model
from src.CNN import MnistCNN
import config

def train(model, loader, optimizer, criterion, device): # training only
    model.train()
    total_loss, correct = 0, 0

    for images, labels in loader: # on all batches
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() 
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / len(loader) # trained loss and accuracy
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device): # evaluation only
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for images, labels in loader: # on all batches
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / len(loader) # actual loss and accuracy
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

def run_training(): 
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders()
    model = MnistCNN().to(device)

    # setting training options
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(1, config.EPOCHS + 1 ):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}"
        )

    # Save model
    save_model( model )

    return model, test_loader

if __name__ == "__main__":
    run_training()