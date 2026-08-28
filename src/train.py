import os
import sys
import time

# Ensure project root directory is added to Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.models.twinlitenet import TwinLiteNet
from src.utils.dataset import TwinLiteNetDataset

# ---------------------------------------------------------
# 1. Custom Imbalance Loss Module (TwinLiteLoss)
# ---------------------------------------------------------
class DiceLoss(nn.Module):
    """
    Computes the Dice Coefficient Loss.
    Crucial for highly imbalanced classes like thin lane markings.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1).float()
        
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice_coeff

class TwinLiteLoss(nn.Module):
    """
    Multi-Task Loss combining BCE and Dice Loss.
    Applies a 1.5x weighting multiplier on the lane Dice loss.
    """
    def __init__(self):
        super(TwinLiteLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred_drivable, pred_lanes, target_drivable, target_lanes):
        # Extract binary target channel logit (class 1)
        drivable_logits = pred_drivable[:, 1, :, :] if pred_drivable.shape[1] > 1 else pred_drivable.squeeze(1)
        lane_logits = pred_lanes[:, 1, :, :] if pred_lanes.shape[1] > 1 else pred_lanes.squeeze(1)

        # Drivable Area Loss (BCE + Dice)
        loss_drivable_bce = self.bce(drivable_logits, target_drivable.float())
        loss_drivable_dice = self.dice(drivable_logits, target_drivable)
        loss_drivable = loss_drivable_bce + loss_drivable_dice

        # Lane Line Loss (BCE + 1.5x Weighted Dice)
        loss_lane_bce = self.bce(lane_logits, target_lanes.float())
        loss_lane_dice = self.dice(lane_logits, target_lanes)
        loss_lane = loss_lane_bce + (1.5 * loss_lane_dice)

        return loss_drivable + loss_lane

# ---------------------------------------------------------
# 2. Evaluation Metrics (Intersection over Union / IoU)
# ---------------------------------------------------------
def calculate_iou(pred_logits, targets, threshold=0.5):
    """Calculates Mean IoU for binary segmentation mask evaluation."""
    probs = torch.sigmoid(pred_logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# ---------------------------------------------------------
# 3. Validation Pass
# ---------------------------------------------------------
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    drivable_iou = 0.0
    lane_iou = 0.0

    with torch.no_grad():
        for images, drivable_masks, lane_masks in val_loader:
            images = images.to(device)
            drivable_masks = drivable_masks.to(device)
            lane_masks = lane_masks.to(device)

            pred_drivable, pred_lanes = model(images)
            loss = criterion(pred_drivable, pred_lanes, drivable_masks, lane_masks)
            val_loss += loss.item()

            d_logits = pred_drivable[:, 1, :, :] if pred_drivable.shape[1] > 1 else pred_drivable.squeeze(1)
            l_logits = pred_lanes[:, 1, :, :] if pred_lanes.shape[1] > 1 else pred_lanes.squeeze(1)

            drivable_iou += calculate_iou(d_logits, drivable_masks)
            lane_iou += calculate_iou(l_logits, lane_masks)

    num_batches = max(len(val_loader), 1)
    return val_loss / num_batches, drivable_iou / num_batches, lane_iou / num_batches

# ---------------------------------------------------------
# 4. Main Training Engine
# ---------------------------------------------------------
def train(epochs=30, batch_size=8, learning_rate=1e-3, val_split_ratio=0.10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("           TWINLITENET DUAL-HEAD TRAINING ENGINE")
    print("=" * 65)
    print(f"Target Compute Hardware : {device}")
    if torch.cuda.is_available():
        print(f"GPU Name                 : {torch.cuda.get_device_name(0)}")

    os.makedirs("weights", exist_ok=True)

    # Load Full Processed Dataset
    full_train_dataset = TwinLiteNetDataset(data_dir="data/processed/train", is_train=True)
    val_dataset_disk = TwinLiteNetDataset(data_dir="data/processed/val", is_train=False)

    # Create dynamic validation split from train dataset if needed
    if len(val_dataset_disk) < 100:
        val_size = int(len(full_train_dataset) * val_split_ratio)
        train_size = len(full_train_dataset) - val_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
        print(f"Dataset Allocation       : {train_size} Train | {val_size} Val (Auto-split)")
    else:
        train_dataset = full_train_dataset
        val_dataset = val_dataset_disk
        print(f"Dataset Allocation       : {len(train_dataset)} Train | {len(val_dataset)} Val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize Model, Criterion, Optimizer, Scheduler
    model = TwinLiteNet().to(device)
    criterion = TwinLiteLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float('inf')

    print("-" * 65)
    print(f"Starting Training Run ({epochs} Epochs | Batch Size: {batch_size})...")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d}", leave=False)
        for images, drivable_masks, lane_masks in pbar:
            images = images.to(device)
            drivable_masks = drivable_masks.to(device)
            lane_masks = lane_masks.to(device)

            optimizer.zero_grad()
            pred_drivable, pred_lanes = model(images)
            
            loss = criterion(pred_drivable, pred_lanes, drivable_masks, lane_masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        scheduler.step()

        epoch_train_loss = running_loss / len(train_loader)
        val_loss, d_iou, l_iou = validate(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Drivable mIoU: {d_iou:.4f} | "
              f"Lane mIoU: {l_iou:.4f}")

        # Save Best Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join("weights", "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'drivable_iou': d_iou,
                'lane_iou': l_iou,
            }, checkpoint_path)
            print(f"  --> Saved new best checkpoint: '{checkpoint_path}' (Val Loss: {val_loss:.4f})")

    # Save Final Epoch Weights
    latest_path = os.path.join("weights", "latest_model.pth")
    torch.save(model.state_dict(), latest_path)
    print("=" * 65)
    print(f"Training Complete! Saved final model to '{latest_path}'")
    print("=" * 65)

if __name__ == "__main__":
    train(epochs=5, batch_size=8, learning_rate=1e-3)