# Dynamic Multi-Task Learning with Gated Routing on IDRiD Dataset
I tried to implement a **gated dynamic routing architecture** for jointly learning **segmentation** and **classification** tasks using the [IDRiD dataset](https://idrid.grand-challenge.org/). The pipeline tries to route inputs to task-specific experts via a learnable gating mechanism.

---

# Model Architecture

The model consists of:
- A **shared ConvNeXt backbone** for feature extraction.
- A lightweight **Gating Network** that predicts the task type (segmentation vs classification) per sample.
- Two **expert heads**:
  - **Segmentation Head**: A shallow U-Net decoder with 4 upsampling blocks to predict 5 retinal lesion masks.
  - **Classification Head**: Two parallel linear classifiers for retinopathy (5 classes) and edema (3 classes).

Each input is routed dynamically during training/inference based on the gate's prediction.

---

# Dataset & Metadata

The [IDRiD dataset](https://idrid.grand-challenge.org/) is split into:
- Segmentation data (`.png` images + masks for 5 lesions).
- Classification data (retino & edema labels in `.csv`).

We use a **balanced metadata JSON** (`balanced_idrid_metadata.json`) to:
- Store image paths, task vectors, class labels, and mask paths.
- Ensure balanced sampling during training.
- Support unified `DataLoader` for both tasks.

---

# Training Strategy

- **Dynamic Routing**: Each image passes through the shared encoder. The **gating network** decides whether it’s routed to segmentation or classification head.
- **Losses**:
  - Segmentation: `BCE + Dice`
  - Classification: `CrossEntropy` for both heads
  - Gating: `CrossEntropy` (supervised routing) + Entropy Regularization
- **Weighted loss contribution** based on task presence in batch.
- **Training Enhancements**:
  - Color jitter, CLAHE
  - Per-batch & per-epoch logging
  - Epoch-wise loss/accuracy tracking

---

##  Evaluation

- **Classification**:
  - Accuracy, Precision, Recall, F1 (macro)
  - Confusion matrix visualized
- **Segmentation**:
  - Per-class Dice & IoU
  - Bar plots for visual comparison
- **Gating Network**:
  - Accuracy per batch and epoch
  - Misrouting detection
- **Visualization**:
  - Overlay predictions vs ground truth
  - Highlights where routing was correct/incorrect

---

##  Inference

- Predicts 20 random samples from test set.
- Displays:
  - Raw image
  - Gated task (cls/seg)
  - Prediction vs GT (masks overlay or label text)
  - Highlights misrouted samples

---


