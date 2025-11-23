# vision/train_classifier.py
"""
Train a simple classifier on synthetic images using PyTorch.

Usage:
    python vision/train_classifier.py

Ensure you have generated synthetic images first:
    python vision/synth_generator.py
"""

import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def train(data_dir='synth_images', epochs=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Transform pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Dataset (expects structure: synth_images/class_x/*.png)
    # If no class folders exist, treat all images as one class
    if not any(os.path.isdir(os.path.join(data_dir, d)) for d in os.listdir(data_dir)):
        # auto-wrap into a single class folder
        tmp_dir = os.path.join(data_dir, "default_class")
        os.makedirs(tmp_dir, exist_ok=True)
        for f in os.listdir(data_dir):
            if f.endswith('.png'):
                os.rename(os.path.join(data_dir, f), os.path.join(tmp_dir, f))

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, len(dataset.classes))
    model = model.to(device)

    # Optimizer
    opt = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for e in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = nn.CrossEntropyLoss()(out, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"Epoch {e+1}/{epochs} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), 'classifier.pth')
    print("Model saved as classifier.pth")


if __name__ == '__main__':
    train()
