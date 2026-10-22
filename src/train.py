from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from safetensors.torch import save_file


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "dataset" / "human detection dataset"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 0.0001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")
print(f"Dataset: {DATA_DIR}")


# --------------------------------------------------
# Image transformations
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# Dataset
# --------------------------------------------------

base_dataset = datasets.ImageFolder(DATA_DIR)

if len(base_dataset.classes) != 2:
    raise ValueError(
        f"Expected two classes, found: {base_dataset.classes}"
    )

print("\nClass mapping:")
for index, name in enumerate(base_dataset.classes):
    print(f"  {index}: {name}")

targets = np.array(base_dataset.targets)
indices = np.arange(len(base_dataset))

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=targets,
)

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=SEED,
    stratify=targets[temp_indices],
)

print("\nDataset split:")
print(f"  Training:   {len(train_indices)}")
print(f"  Validation: {len(val_indices)}")
print(f"  Testing:    {len(test_indices)}")


# Use separate dataset objects so training augmentation
# does not affect validation and test images.
train_dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=train_transform,
)

eval_dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=eval_transform,
)

train_loader = DataLoader(
    Subset(train_dataset, train_indices.tolist()),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    Subset(eval_dataset, val_indices.tolist()),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    Subset(eval_dataset, test_indices.tolist()),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# --------------------------------------------------
# Model
# --------------------------------------------------

print("\nLoading pretrained ResNet-18...")

try:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
except Exception as error:
    raise RuntimeError(
        "Could not load pretrained ResNet-18 weights. "
        "Check your internet connection and try again."
    ) from error

# Freeze the pretrained feature extractor.
for parameter in model.parameters():
    parameter.requires_grad = False

# Train only the final classification layer.
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE,
)


# --------------------------------------------------
# Evaluation helper
# --------------------------------------------------

def evaluate(loader):
    model.eval()

    all_labels = []
    all_predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels, all_predictions
    )

    return average_loss, accuracy, all_labels, all_predictions


# --------------------------------------------------
# Training
# --------------------------------------------------

best_val_loss = float("inf")
best_weights = None
patience = 3
epochs_without_improvement = 0

print("\nStarting training...\n")

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    val_loss, val_accuracy, _, _ = evaluate(val_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Validation loss: {val_loss:.4f} | "
        f"Validation accuracy: {val_accuracy:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        # Copy tensors to ordinary CPU tensors for export.
        best_weights = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in model.state_dict().items()
        }

        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping: validation loss did not improve.")
        break


# --------------------------------------------------
# Save best model
# --------------------------------------------------

if best_weights is None:
    raise RuntimeError("Training did not produce model weights.")

weights_path = MODEL_DIR / "human_classifier.safetensors"

save_file(best_weights, str(weights_path))

config = {
    "architecture": "resnet18",
    "num_classes": 2,
    "class_to_idx": base_dataset.class_to_idx,
    "image_size": IMAGE_SIZE,
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "best_validation_loss": best_val_loss,
}

config_path = MODEL_DIR / "human_classifier_config.json"

with open(config_path, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=4)


# --------------------------------------------------
# Test best model
# --------------------------------------------------

model.load_state_dict(best_weights)

test_loss, test_accuracy, y_true, y_pred = evaluate(test_loader)

print("\nFinal test results")
print("-" * 40)
print(f"Test loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(base_dataset.classes))),
        target_names=base_dataset.classes,
        zero_division=0,
    )
)

print(f"\nSafetensors weights: {weights_path}")
print(f"Model configuration: {config_path}")
print("\nTraining completed.")