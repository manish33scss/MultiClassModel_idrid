#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 09:54:55 2025

@author: Manish
"""

# only classification

import json
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# ---------------------------------------
# 1. Load JSON and Filter Classification Tasks
# ---------------------------------------
with open("/content/drive/MyDrive/idrid/balanced_idrid_metadata.json", "r") as f:
    metadata = json.load(f)

cls_entries = [entry for entry in metadata if entry['task'] == 'cls']

# ---------------------------------------
# 2. Custom Dataset for Classification
# ---------------------------------------
class IDRiDClassificationDataset(Dataset):
    def __init__(self, entries, transform=None):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image = Image.open(entry["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        retino_label = entry["retino_label"]
        edema_label = entry["edema_label"]
        return image, torch.tensor([retino_label, edema_label], dtype=torch.long)

# ---------------------------------------
# 3. Transforms and Dataloaders
# ---------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = IDRiDClassificationDataset(cls_entries, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

# ---------------------------------------
# 4. Model Definition: ConvNeXt + 2 Heads
# ---------------------------------------
class MultiTaskClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.convnext_tiny(pretrained=True)
        self.backbone.classifier = nn.Identity()  # remove default head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_retino = nn.Linear(768, 5)  # 5 classes
        self.fc_edema = nn.Linear(768, 3)   # 3 classes

    def forward(self, x):
        x = self.backbone.features(x)  # (B, 768, H, W)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B, 768)
        return self.fc_retino(x), self.fc_edema(x)

# ---------------------------------------
# 5. Training Setup
# ---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskClassifier().to(device)

criterion_retino = nn.CrossEntropyLoss()
criterion_edema = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# ---------------------------------------
# 6. Training Loop
# ---------------------------------------
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        retino_logits, edema_logits = model(images)

        loss_retino = criterion_retino(retino_logits, labels[:, 0])
        loss_edema = criterion_edema(edema_logits, labels[:, 1])
        loss = loss_retino + loss_edema

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

# ---------------------------------------
# 7. Save Model
# ---------------------------------------
torch.save(model.state_dict(), "convnext_cls_retino_edema.pth")
print("Model saved!")

# ---------------------------------------
# 8. Evaluation
# ---------------------------------------
from sklearn.metrics import accuracy_score

model.eval()
all_preds_retino = []
all_preds_edema = []
all_labels_retino = []
all_labels_edema = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        retino_logits, edema_logits = model(images)
        pred_retino = torch.argmax(retino_logits, dim=1)
        pred_edema = torch.argmax(edema_logits, dim=1)

        all_preds_retino.extend(pred_retino.cpu().numpy())
        all_preds_edema.extend(pred_edema.cpu().numpy())
        all_labels_retino.extend(labels[:, 0].cpu().numpy())
        all_labels_edema.extend(labels[:, 1].cpu().numpy())

print(f"📊 Retinopathy Accuracy: {accuracy_score(all_labels_retino, all_preds_retino):.4f}")
print(f"📊 Edema Accuracy:       {accuracy_score(all_labels_edema, all_preds_edema):.4f}")
