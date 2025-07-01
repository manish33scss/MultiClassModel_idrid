#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 09:49:13 2025

@author: Manish
"""

import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms

# ----------------------------
# 1. Define the Model
# ----------------------------
from torchvision import models

class UpBlock(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ConvNeXtUNet(torch.nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.encoder = models.convnext_tiny(pretrained=True)
        self.encoder.classifier = torch.nn.Identity()
        self.pool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.up1 = UpBlock(768, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 32)
        self.final = torch.nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder.features(x)
        x = self.pool(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.final(x)

# ----------------------------
# 2. Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvNeXtUNet(num_classes=5)
model.load_state_dict(torch.load("convnext_segmentation.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

mask_keys = ["MA", "HE", "EX", "SE", "OD"]
seg_colors = {
    "MA": (0, 0, 255),      # Red
    "HE": (0, 255, 0),      # Green
    "EX": (255, 0, 0),      # Blue
    "SE": (0, 255, 255),    # Yellow
    "OD": (255, 0, 255)     # Magenta
}
class_descriptions = {
    "MA": "Microaneurysms",
    "HE": "Haemorrhages",
    "EX": "Hard Exudates",
    "SE": "Soft Exudates",
    "OD": "Optic Disc"
}

# ----------------------------
# 3. Visualization Function
# ----------------------------
def visualize_sample_with_prediction(sample):
    image_path = sample["image_path"]
    mask_dict = sample["mask_paths"]

    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    image = cv2.resize(image, (224, 224))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(Image.fromarray(image_rgb)).unsqueeze(0).to(device)

    # Predict masks
    with torch.no_grad():
        pred = torch.sigmoid(model(image_tensor))[0]  # (5, 224, 224)

    # Prepare overlays
    gt_overlay = image_rgb.copy()
    pred_overlay = image_rgb.copy()
    mask_display = np.zeros_like(image_rgb)

    for idx, class_key in enumerate(mask_keys):
        seg_path = mask_dict.get(class_key)
        color = seg_colors[class_key][::-1]  # BGR to RGB

        # Ground truth overlay
        if seg_path and Path(seg_path).exists():
            gt_mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                gt_mask = cv2.resize(gt_mask, (224, 224))
                _, binary_mask = cv2.threshold(gt_mask, 1, 255, cv2.THRESH_BINARY)
                color_mask = np.zeros_like(image_rgb)
                color_mask[binary_mask > 0] = color
                gt_overlay = cv2.addWeighted(color_mask, 0.5, gt_overlay, 1, 0)
                mask_display[binary_mask > 0] = color

        # Prediction overlay
        pred_mask = (pred[idx] > 0.5).cpu().numpy().astype(np.uint8)
        color_mask = np.zeros_like(image_rgb)
        for j in range(3):
            color_mask[:, :, j] = pred_mask * color[j]
        pred_overlay = cv2.addWeighted(color_mask, 0.5, pred_overlay, 1, 0)

    # Plot
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gt_overlay)
    plt.title("Ground Truth Overlay")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_overlay)
    plt.title("Prediction Overlay")
    plt.axis('off')

    # Legend
    legend_elements = []
    for seg_type, color in seg_colors.items():
        rgb_color = np.array(color[::-1]) / 255.0
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', label=class_descriptions[seg_type],
                       markerfacecolor=rgb_color, markersize=10)
        )
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# ----------------------------
# 4. Run on Your JSON
# ----------------------------
# Load metadata JSON
json_path = "/content/drive/MyDrive/idrid/segmentation_metadata_train.json"  # update if needed
with open(json_path, 'r') as f:
    data = json.load(f)

# Visualize first few samples
for sample in data[:5]:
    print(f"\n▶ Visualizing: {Path(sample['image_path']).name}")
    visualize_sample_with_prediction(sample)









# Load your updated JSON
with open("/content/drive/MyDrive/idrid/segmentation_metadata_train.json") as f:
    data = json.load(f)

# Visualize a few samples
for sample in data[:5]:  # or random.sample(data, 5)
    print(f"▶ Visualizing: {Path(sample['image_path']).name}")
    visualize_sample_with_prediction(sample)
