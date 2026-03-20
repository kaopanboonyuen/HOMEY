# ===============================================================
#  🏠 HOMEY: Heuristic Object Masking with Enhanced YOLO
# ===============================================================
#  Author       : Teerapong Panboonyuen
#  Affiliation  : MARSAIL (Motor AI Recognition Solution Artificial Intelligence Laboratory)
# ===============================================================
#  Script       : <FILENAME.py>
#  Description  : 
#      This script is part of the HOMEY framework for property
#      insurance risk detection. HOMEY integrates YOLOv8 backbone
#      with heuristic object masking and risk-aware loss calibration
#      to detect structural damages, maintenance neglect, and 
#      liability hazards across 17 property risk classes.
# ===============================================================
#  Python       : >=3.10
#  PyTorch      : >=2.1
#  Ultralytics YOLO : >=8.1
# ===============================================================
#  License      : MIT License
#  Citation     : 
#      T. Panboonyuen, "HOMEY: Heuristic Object Masking with 
#      Enhanced YOLO for Property Insurance Risk Detection", 
#      arXiv:2603.18502, 2026.
# ===============================================================


import torch
from torch.utils.data import DataLoader
from models.homey_model import HOMEYModel
from datasets.property_dataset import PropertyDataset
from losses.risk_loss import RiskAwareLoss
import os

# -----------------------------
# Configuration
# -----------------------------
train_dir = "../data/train"
val_dir = "../data/val"
num_classes = 17
batch_size = 8
epochs = 50
lr = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "../outputs"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Dataset & Dataloader
# -----------------------------
train_dataset = PropertyDataset(train_dir)
val_dataset = PropertyDataset(val_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# Model, Loss, Optimizer
# -----------------------------
model = HOMEYModel(num_classes=num_classes).to(device)
criterion = RiskAwareLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {total_loss/len(train_loader):.4f}")

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            val_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}] - Val Loss: {val_loss/len(val_loader):.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(output_dir, f"homey_epoch{epoch+1}.pth"))

print("Training Complete ✅")