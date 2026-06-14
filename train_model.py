import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomRotation(15),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.3), ratio=(0.3, 3.3))
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset
train_dataset = datasets.ImageFolder("dataset/train", transform=train_transform)
val_dataset   = datasets.ImageFolder("dataset/val",   transform=val_transform)

# Verify mapping
print("Class mapping:", train_dataset.class_to_idx)
print("Train size:", len(train_dataset), "| Val size:", len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

# Model
model = models.mobilenet_v3_small(weights="DEFAULT")
for param in model.features.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(
    nn.Linear(576, 128),
    nn.Hardswish(),
    nn.Dropout(0.2),
    nn.Linear(128, 5)
)

model = model.to(device)

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training config
epochs        = 40
best_acc      = 0
patience      = 7       # early stopping
no_improve    = 0
unfreeze_done = False

# Training Loop
for epoch in range(epochs):

    # unfreeze backbone at epoch 10
    if epoch == 10 and not unfreeze_done:
        print("\nUnfreezing backbone for fine-tuning...")
        for param in model.features.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        unfreeze_done = True

    # Train
    model.train()
    running_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = running_loss / len(train_loader)

    # Validate
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (pred == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"\nEpoch {epoch+1}/{epochs}  |  Loss: {train_loss:.4f}  |  Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_acc:
        best_acc   = val_acc
        no_improve = 0
        torch.save(model.state_dict(), "best_model_temp.pth")
        print(f"  --> Best model saved ({val_acc:.2f}%)")
    else:
        no_improve += 1
        print(f"  --> No improvement ({no_improve}/{patience})")

    # Early stopping
    if no_improve >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

    scheduler.step()

# Save final model
model.load_state_dict(torch.load("best_model_temp.pth"))
model.eval().to("cpu")
torch.save(model.state_dict(), "student_model.pth")
os.remove("best_model_temp.pth")
size_mb = os.path.getsize("student_model.pth") / 1e6
print(f"\nBest Validation Accuracy : {best_acc:.2f}%")
print(f"Final model size         : {size_mb:.2f} MB")