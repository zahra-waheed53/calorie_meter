import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from time import time

CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "foodnet.pth")

EPOCHS = 5
BATCH_SIZE = 16
LR = 0.001

def train_model():

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder("dataset/train", transform=transform_train)
    val_dataset = datasets.ImageFolder("dataset/val", transform=transform_val)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    num_classes = len(train_dataset.classes)

    # -------- MobileNetV2 --------
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")

    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )

    device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

    print("\n🚀 Training started (CPU optimized)")
    start_time = time()

    for epoch in range(EPOCHS):
        epoch_start = time()

        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        acc = 100 * correct / total
        epoch_time = (time() - epoch_start) / 60

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {avg_loss:.4f} "
            f"Val Acc: {acc:.2f}% "
            f"Time: {epoch_time:.1f} min"
        )

    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": train_dataset.classes,
        },
        MODEL_PATH,
    )

    total_time = (time() - start_time) / 60
    print(f"\n✅ Training complete in {total_time:.1f} minutes")
    print(f"💾 Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
