import os
import sys
import time

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Ensure project root directory is added to Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.models.twinlitenet import TwinLiteNet
from src.utils.dataset import TwinLiteNetDataset


def extract_binary_logits(pred_tensor):
    """
    Extracts unified binary logits from 1-channel or 2-channel network outputs.
    For 2-channel outputs [B, 2, H, W], sigmoid(z1 - z0) == softmax(z)[1],
    ensuring both channels receive gradients and match infer.py softmax evaluation.
    """
    if pred_tensor.shape[1] > 1:
        return pred_tensor[:, 1, :, :] - pred_tensor[:, 0, :, :]
    return pred_tensor.squeeze(1)


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
        targets = targets.float()
        
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
    def __init__(self, lane_pos_weight=3.0):
        super(TwinLiteLoss, self).__init__()
        self.bce_drivable = nn.BCEWithLogitsLoss(reduction='none')
        # Positive weight boosts recall on thin lane markings (<1.5% of image pixels)
        self.register_buffer('pos_weight', torch.tensor([lane_pos_weight]))
        self.dice = DiceLoss()

    def forward(self, pred_drivable, pred_lanes, target_drivable, target_lanes, lane_valids=None):
        drivable_logits = extract_binary_logits(pred_drivable)
        lane_logits = extract_binary_logits(pred_lanes)

        # ---------------------------------------------------
        # Drivable Area Loss (Supervised across valid masks)
        # ---------------------------------------------------
        drivable_valid_mask = (target_drivable.sum(dim=(1, 2)) > 0).float()
        loss_d_bce_unreduced = self.bce_drivable(drivable_logits, target_drivable.float()).mean(dim=(1, 2))
        loss_d_dice_unreduced = self.dice(drivable_logits, target_drivable, reduce=False)
        loss_d_unreduced = loss_d_bce_unreduced + loss_d_dice_unreduced

        d_valid_count = torch.clamp(drivable_valid_mask.sum(), min=1.0)
        loss_drivable = (loss_d_unreduced * drivable_valid_mask).sum() / d_valid_count

        # ---------------------------------------------------
        # Lane Line Loss (Domain-Gated via Selective Masking)
        # ---------------------------------------------------
        pos_w = self.pos_weight.to(lane_logits.device)
        loss_lane_bce_unreduced = nn.functional.binary_cross_entropy_with_logits(
            lane_logits, target_lanes.float(), pos_weight=pos_w, reduction='none'
        ).mean(dim=(1, 2))
        loss_lane_dice_unreduced = self.dice(lane_logits, target_lanes, reduce=False)
        loss_lane_unreduced = loss_lane_bce_unreduced + (1.5 * loss_lane_dice_unreduced)

        if lane_valids is not None:
            masked_lane_loss = loss_lane_unreduced * lane_valids
            valid_count = torch.clamp(lane_valids.sum(), min=1.0)
            loss_lane = masked_lane_loss.sum() / valid_count
        else:
            loss_lane = loss_lane_unreduced.mean()

        return loss_drivable + loss_lane


# ---------------------------------------------------------
# 2. Evaluation Metrics (Domain-Aware IoU)
# ---------------------------------------------------------
def calculate_iou(pred_logits, targets, valid_flags=None, threshold=0.5):
    """
    Calculates Mean IoU strictly over valid annotated samples in the batch.
    Returns (iou_sum, valid_sample_count) for accurate dataset-wide averaging.
    """
    probs = torch.sigmoid(pred_logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) - intersection
    iou_per_image = (intersection + 1e-6) / (union + 1e-6)

    # Only evaluate IoU on images that actually have ground-truth annotations
    has_gt = (targets.sum(dim=(1, 2)) > 0).float()
    if valid_flags is not None:
        mask = has_gt * (valid_flags > 0.5).float()
    else:
        mask = has_gt

    valid_count = mask.sum().item()
    if valid_count == 0:
        return 0.0, 0

    iou_sum = (iou_per_image * mask).sum().item()
    return iou_sum, valid_count


# ---------------------------------------------------------
# 3. Validation Pass
# ---------------------------------------------------------
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    
    drivable_iou_sum = 0.0
    drivable_valid_total = 0
    
    lane_iou_sum = 0.0
    lane_valid_total = 0

    with torch.no_grad():
        for images, drivable_masks, lane_masks, lane_valids in val_loader:
            images = images.to(device)
            drivable_masks = drivable_masks.to(device)
            lane_masks = lane_masks.to(device)
            lane_valids = lane_valids.to(device)

            with torch.amp.autocast('cuda'):
                pred_drivable, pred_lanes = model(images)
                loss = criterion(pred_drivable, pred_lanes, drivable_masks, lane_masks, lane_valids)
            
            val_loss += loss.item()

            d_logits = extract_binary_logits(pred_drivable)
            l_logits = extract_binary_logits(pred_lanes)

            d_sum, d_cnt = calculate_iou(d_logits, drivable_masks)
            drivable_iou_sum += d_sum
            drivable_valid_total += d_cnt

            l_sum, l_cnt = calculate_iou(l_logits, lane_masks, valid_flags=lane_valids)
            lane_iou_sum += l_sum
            lane_valid_total += l_cnt

    num_batches = max(len(val_loader), 1)
    avg_val_loss = val_loss / num_batches
    avg_d_iou = drivable_iou_sum / max(drivable_valid_total, 1)
    avg_l_iou = lane_iou_sum / max(lane_valid_total, 1)

    return avg_val_loss, avg_d_iou, avg_l_iou


# ---------------------------------------------------------
# 4. Main Training Engine
# ---------------------------------------------------------
def train(
    additional_epochs=10,
    batch_size=8,
    learning_rate=1e-4,
    val_split_ratio=0.10,
    resume_checkpoint=None,
    reset_best_loss=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("           TWINLITENET DUAL-HEAD TRAINING ENGINE")
    print("=" * 65)
    print(f"Target Compute Hardware : {device}")
    if torch.cuda.is_available():
        print(f"GPU Name                : {torch.cuda.get_device_name(0)}")

    os.makedirs("twin/weights", exist_ok=True)

    full_train_dataset = TwinLiteNetDataset(data_dir="data/processed/train", is_train=True)
    val_dataset_disk = TwinLiteNetDataset(data_dir="data/processed/val", is_train=False)

    if len(val_dataset_disk) < 100:
        val_size = int(len(full_train_dataset) * val_split_ratio)
        train_size = len(full_train_dataset) - val_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
        print(f"Dataset Allocation      : {train_size} Train | {val_size} Val (Auto-split)")
    else:
        train_dataset = full_train_dataset
        val_dataset = val_dataset_disk
        print(f"Dataset Allocation      : {len(train_dataset)} Train | {len(val_dataset)} Val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = TwinLiteNet().to(device)
    criterion = TwinLiteLoss(lane_pos_weight=3.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    current_epoch = 0

    # Resume from checkpoint if specified
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming weights from   : '{resume_checkpoint}'")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            current_epoch = checkpoint.get('epoch', 0)
            if not reset_best_loss:
                best_val_loss = checkpoint.get('val_loss', float('inf'))
        else:
            model.load_state_dict(checkpoint)

        # Force the optimizer to use the fresh target learning_rate for the new schedule
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
            param_group['initial_lr'] = learning_rate

    start_epoch = current_epoch + 1
    total_target_epochs = current_epoch + additional_epochs

    # Schedule strictly over the additional_epochs so LR starts at learning_rate and decays to eta_min
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=additional_epochs, eta_min=1e-5)

    print("-" * 65)
    print(f"Starting Training Run (Adding {additional_epochs} Epochs -> Targeting Epoch {total_target_epochs} | Batch Size: {batch_size})...")
    print("-" * 65)

    try:
        for epoch in range(start_epoch, total_target_epochs + 1):
            model.train()
            running_loss = 0.0
            start_time = time.time()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{total_target_epochs:02d}", leave=False)
            
            for images, drivable_masks, lane_masks, lane_valids in pbar:
                images = images.to(device)
                drivable_masks = drivable_masks.to(device)
                lane_masks = lane_masks.to(device)
                lane_valids = lane_valids.to(device)

                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    pred_drivable, pred_lanes = model(images)
                    loss = criterion(pred_drivable, pred_lanes, drivable_masks, lane_masks, lane_valids)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

            # Step scheduler AFTER optimizer steps (fixes PyTorch lr_scheduler warning)
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
    # Resume from latest_model.pth (Epoch 25 weights) and reset best_val_loss so best_model.pth updates
    train(
        additional_epochs=5,
        batch_size=8,
        learning_rate=1e-4,
        resume_checkpoint="twin/weights/latest_model.pth",
        reset_best_loss=True,
    )