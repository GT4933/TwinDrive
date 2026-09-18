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
    Computes the Dice Coefficient Loss per sample or across the batch.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, reduce=True):
        probs = torch.sigmoid(logits)
        
        # Calculate intersection and union per batch item (dim 1 and 2 are H and W)
        intersection = (probs * targets).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1. - dice_coeff
        
        if reduce:
            return loss.mean()
        return loss

class TwinLiteLoss(nn.Module):
    """
    Multi-Task Loss with Domain-Gated Masking for Unlabeled Lanes.
    """
    def __init__(self):
        super(TwinLiteLoss, self).__init__()
        # reduction='none' allows us to isolate loss per image in the batch
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dice = DiceLoss()

    def forward(self, pred_drivable, pred_lanes, target_drivable, target_lanes, lane_valids=None):
        drivable_logits = pred_drivable[:, 1, :, :] if pred_drivable.shape[1] > 1 else pred_drivable.squeeze(1)
        lane_logits = pred_lanes[:, 1, :, :] if pred_lanes.shape[1] > 1 else pred_lanes.squeeze(1)

        # ---------------------------------------------------
        # Drivable Area Loss (Supervised across ALL frames)
        # ---------------------------------------------------
        loss_drivable_bce = self.bce(drivable_logits, target_drivable.float()).mean()
        loss_drivable_dice = self.dice(drivable_logits, target_drivable, reduce=True)
        loss_drivable = loss_drivable_bce + loss_drivable_dice

        # ---------------------------------------------------
        # Lane Line Loss (Domain-Gated via Selective Masking)
        # ---------------------------------------------------
        # 1. Calculate unreduced loss (Shape: [Batch_Size])
        loss_lane_bce_unreduced = self.bce(lane_logits, target_lanes.float()).mean(dim=(1, 2))
        loss_lane_dice_unreduced = self.dice(lane_logits, target_lanes, reduce=False)
        loss_lane_unreduced = loss_lane_bce_unreduced + (1.5 * loss_lane_dice_unreduced)

        # 2. Apply the domain filter
        if lane_valids is not None:
            masked_lane_loss = loss_lane_unreduced * lane_valids
            # Prevent division by zero if entire batch happens to be IDD
            valid_count = torch.clamp(lane_valids.sum(), min=1.0)
            loss_lane = masked_lane_loss.sum() / valid_count
        else:
            loss_lane = loss_lane_unreduced.mean()

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
        # Unpack the 4th item (lane_valids)
        for images, drivable_masks, lane_masks, lane_valids in val_loader:
            images = images.to(device)
            drivable_masks = drivable_masks.to(device)
            lane_masks = lane_masks.to(device)
            lane_valids = lane_valids.to(device)

            # Updated AMP syntax to remove deprecation warning
            with torch.amp.autocast('cuda'):
                pred_drivable, pred_lanes = model(images)
                loss = criterion(pred_drivable, pred_lanes, drivable_masks, lane_masks, lane_valids)
            
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
def train(additional_epochs=5, batch_size=8, learning_rate=1e-4, val_split_ratio=0.10, resume_checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("           TWINLITENET DUAL-HEAD TRAINING ENGINE")
    print("=" * 65)
    print(f"Target Compute Hardware : {device}")
    if torch.cuda.is_available():
        print(f"GPU Name                 : {torch.cuda.get_device_name(0)}")

    # Route weights directly to the twin subfolder
    os.makedirs("twin/weights", exist_ok=True)

    full_train_dataset = TwinLiteNetDataset(data_dir="data/processed/train", is_train=True)
    val_dataset_disk = TwinLiteNetDataset(data_dir="data/processed/val", is_train=False)

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

    model = TwinLiteNet().to(device)
    criterion = TwinLiteLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Initialize AMP Scaler (Updated syntax to fix warning)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    current_epoch = 0

    # Resume from checkpoint if specified
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming weights from    : '{resume_checkpoint}'")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('val_loss', float('inf'))

    start_epoch = current_epoch + 1
    total_target_epochs = current_epoch + additional_epochs

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_target_epochs, eta_min=1e-5)
    
    # Fast-forward scheduler to match the current epoch
    for _ in range(current_epoch):
        scheduler.step()

    print("-" * 65)
    print(f"Starting Training Run (Adding {additional_epochs} Epochs -> Targeting Epoch {total_target_epochs} | Batch Size: {batch_size})...")
    print("-" * 65)

    try:
        for epoch in range(start_epoch, total_target_epochs + 1):
            model.train()
            running_loss = 0.0
            start_time = time.time()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{total_target_epochs:02d}", leave=False)
            
            # Unpack the 4th item (lane_valids)
            for images, drivable_masks, lane_masks, lane_valids in pbar:
                images = images.to(device)
                drivable_masks = drivable_masks.to(device)
                lane_masks = lane_masks.to(device)
                lane_valids = lane_valids.to(device)

                optimizer.zero_grad()
                
                # Cast forward pass to mixed precision (Updated syntax)
                with torch.amp.autocast('cuda'):
                    pred_drivable, pred_lanes = model(images)
                    loss = criterion(pred_drivable, pred_lanes, drivable_masks, lane_masks, lane_valids)
                
                # Scaled backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

            scheduler.step()

            epoch_train_loss = running_loss / len(train_loader)
            val_loss, d_iou, l_iou = validate(model, val_loader, criterion, device)
            elapsed = time.time() - start_time

            print(f"Epoch [{epoch:02d}/{total_target_epochs:02d}] ({elapsed:.1f}s) | "
                  f"Train Loss: {epoch_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Drivable mIoU: {d_iou:.4f} | "
                  f"Lane mIoU: {l_iou:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join("twin", "weights", "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'drivable_iou': d_iou,
                    'lane_iou': l_iou,
                }, checkpoint_path)
                print(f"  --> Saved new best checkpoint: '{checkpoint_path}' (Val Loss: {val_loss:.4f})")
    
    except KeyboardInterrupt:
        print("\n[!] Training manually interrupted. Existing checkpoints remain intact.")

    latest_path = os.path.join("twin", "weights", "latest_model.pth")
    torch.save(model.state_dict(), latest_path)
    print("=" * 65)
    print(f"Training Complete! Saved final model to '{latest_path}'")
    print("=" * 65)

if __name__ == "__main__":
    train(additional_epochs=5, batch_size=8, learning_rate=1e-4, resume_checkpoint="twin/weights/best_model.pth")