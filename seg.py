#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 09:48:44 2025

@author: Manish
"""

import json
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------
# 1. Load JSON and Extract Segmentation Entries
# ---------------------------------------
with open("/content/drive/MyDrive/idrid/balanced_idrid_metadata.json", "r") as f:
    metadata = json.load(f)

seg_entries = [entry for entry in metadata if entry['task'] == 'seg']
mask_keys = ["MA", "HE", "EX", "SE", "OD"]
num_classes = len(mask_keys)

# ---------------------------------------
# 2. Custom Dataset for Segmentation
# ---------------------------------------
class IDRiDSegmentationDataset(Dataset):
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

        masks = []
        for key in mask_keys:
            mask_path = entry["mask_paths"].get(key)
            if mask_path is not None:
                mask = Image.open(mask_path).resize((224, 224))
                mask = np.array(mask) > 0  # binary mask
            else:
                mask = np.zeros((224, 224), dtype=np.uint8)
            masks.append(mask.astype(np.float32))

        masks = np.stack(masks, axis=0)  # shape: (5, H, W)
        return image, torch.tensor(masks)

# ---------------------------------------
# 3. Transforms and Dataloaders
# ---------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = IDRiDSegmentationDataset(seg_entries, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

# ---------------------------------------
# 4. Model: ConvNeXt + U-Net Style Decoder
# ---------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ConvNeXtUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = models.convnext_tiny(pretrained=True)
        self.encoder.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))  # match feature map size

        self.up1 = UpBlock(768, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 32)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder.features(x)
        x = self.pool(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.final(x)

# ---------------------------------------
# 5. Loss Functions: BCE + Dice
# ---------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + self.eps) / (union + self.eps)
        dice_loss = 1 - dice.mean(dim=1)
        return dice_loss.mean()

bce_loss = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss()

def combo_loss(preds, targets):
    return bce_loss(preds, targets) + dice_loss(preds, targets)

# ---------------------------------------
# 6. Training Loop
# ---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNeXtUNet(num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
epochs = 5
train_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, masks = images.to(device), masks.to(device)
        preds = model(images)

        loss = combo_loss(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

# ---------------------------------------
# 7. Save Model
# ---------------------------------------
torch.save(model.state_dict(), "convnext_segmentation.pth")
print("Segmentation model saved!")

# ---------------------------------------
# 8. Evaluation: Per-Class Dice Score
# ---------------------------------------
def dice_per_class(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred) > 0.5
    dice_scores = []
    for i in range(pred.shape[1]):
        p = pred[:, i]
        t = target[:, i]
        intersection = (p * t).sum(dim=(1, 2))
        union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        dice = (2 * intersection + eps) / (union + eps)
        dice_scores.append(dice.mean().item())
    return dice_scores

model.eval()
dice_totals = [0.0] * num_classes
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        preds = model(images)
        scores = dice_per_class(preds, masks)
        for i in range(num_classes):
            dice_totals[i] += scores[i]

avg_dice = [x / len(test_loader) for x in dice_totals]
print("\n📊 Per-Class Dice Scores:")
for name, score in zip(mask_keys, avg_dice):
    print(f"{name}: {score:.4f}")

# ---------------------------------------
# 9. Plot Loss Curve
# ---------------------------------------
plt.plot(train_losses, label="Training Loss")
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()
