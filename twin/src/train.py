import os
import sys
import csv
import time
import argparse

# One math thread per process: each DataLoader worker otherwise starts a full OpenBLAS/OpenMP thread pool
# with its own buffers, which exhausted RAM on a 16 GB Windows machine. Must be set before numpy/torch load.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

# Ensure project root directory is added to Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.models.twinlitenet import TwinLiteNet
from src.utils.dataset import TwinLiteNetDataset
from src.utils.metrics import BinarySegMetric

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
        # Heads output (B, 1, H, W) logits; drop the channel dim to match (B, H, W) targets
        drivable_logits = pred_drivable.squeeze(1)
        lane_logits = pred_lanes.squeeze(1)

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
# 2. Validation Pass (dataset-level metrics, reported per domain)
# ---------------------------------------------------------
def validate(model, val_loader, criterion, device, threshold=0.5):
    """
    Returns (mean val loss, metrics dict). Confusion matrices are accumulated over the whole
    val set and split by domain: lanes are scored on BDD frames only (IDD has no lane labels).
    """
    model.eval()
    val_loss = 0.0
    da_bdd, da_idd, ll_bdd = BinarySegMetric(), BinarySegMetric(), BinarySegMetric()

    with torch.no_grad():
        for images, drivable_masks, lane_masks, lane_valids in val_loader:
            images = images.to(device)
            drivable_masks = drivable_masks.to(device)
            lane_masks = lane_masks.to(device)
            lane_valids = lane_valids.to(device)

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                pred_drivable, pred_lanes = model(images)
                loss = criterion(pred_drivable, pred_lanes, drivable_masks, lane_masks, lane_valids)
            val_loss += loss.item()

            d_pred = torch.sigmoid(pred_drivable.squeeze(1).float()) > threshold
            l_pred = torch.sigmoid(pred_lanes.squeeze(1).float()) > threshold

            is_bdd = lane_valids > 0.5
            if is_bdd.any():
                da_bdd.update(d_pred[is_bdd], drivable_masks[is_bdd])
                ll_bdd.update(l_pred[is_bdd], lane_masks[is_bdd])
            if (~is_bdd).any():
                da_idd.update(d_pred[~is_bdd], drivable_masks[~is_bdd])

    metrics = {
        "bdd_da_miou": da_bdd.miou() if da_bdd.num_pixels() else float("nan"),
        "bdd_ll_iou": ll_bdd.iou() if ll_bdd.num_pixels() else float("nan"),
        "bdd_ll_recall": ll_bdd.recall() if ll_bdd.num_pixels() else float("nan"),
        "idd_da_miou": da_idd.miou() if da_idd.num_pixels() else float("nan"),
    }
    return val_loss / max(len(val_loader), 1), metrics

def selection_score(metrics):
    """Checkpoint selection on what we report: mean of BDD drivable mIoU and BDD lane IoU."""
    return (metrics["bdd_da_miou"] + metrics["bdd_ll_iou"]) / 2.0

# ---------------------------------------------------------
# 3. Main Training Engine
# ---------------------------------------------------------
def subset(dataset, max_samples, seed=0):
    """Fixed random subset for quick smoke tests; returns the full dataset when max_samples is None."""
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    return Subset(dataset, torch.randperm(len(dataset), generator=g)[:max_samples].tolist())

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print("=" * 65)
    print("           TWINLITENET DUAL-HEAD TRAINING ENGINE")
    print("=" * 65)
    print(f"Target Compute Hardware : {device}")
    if torch.cuda.is_available():
        print(f"GPU Name                 : {torch.cuda.get_device_name(0)}")

    os.makedirs(args.out_dir, exist_ok=True)

    train_dataset = subset(TwinLiteNetDataset(data_dir="data/processed/train", is_train=True), args.max_train_samples)
    val_dataset = subset(TwinLiteNetDataset(data_dir="data/processed/val", is_train=False), args.max_val_samples)
    print(f"Dataset Allocation       : {len(train_dataset)} Train | {len(val_dataset)} Val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    # Val workers are re-spawned each epoch and kept few: on Windows, persistent train workers plus a second
    # full set of val workers exhausted memory and hung the run
    val_workers = min(args.workers, 2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=val_workers, pin_memory=True)

    model = TwinLiteNet().to(device)
    criterion = TwinLiteLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # One fixed schedule over the whole run; its state is checkpointed so resuming continues the same curve
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 1
    best_score = -1.0
    latest_path = os.path.join(args.out_dir, "latest_model.pth")
    best_path = os.path.join(args.out_dir, "best_model.pth")
    log_path = os.path.join(args.out_dir, "train_log.csv")

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint '{args.resume}' not found")
        print(f"Resuming from            : '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        saved_epochs = checkpoint['args']['epochs']
        if saved_epochs != args.epochs:
            raise ValueError(f"Checkpoint was trained with --epochs {saved_epochs}; resuming with --epochs "
                             f"{args.epochs} would distort the LR schedule. Pass --epochs {saved_epochs}.")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint.get('best_score', -1.0)
    elif os.path.exists(log_path):
        os.remove(log_path)

    print("-" * 65)
    print(f"Training epochs {start_epoch}..{args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("-" * 65)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running_loss = 0.0
            start_time = time.time()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d}", leave=False)

            for images, drivable_masks, lane_masks, lane_valids in pbar:
                images = images.to(device, non_blocking=True)
                drivable_masks = drivable_masks.to(device, non_blocking=True)
                lane_masks = lane_masks.to(device, non_blocking=True)
                lane_valids = lane_valids.to(device, non_blocking=True)

                optimizer.zero_grad()

                # Cast forward pass to mixed precision
                with torch.amp.autocast(device.type, enabled=use_amp):
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
            val_loss, m = validate(model, val_loader, criterion, device)
            score = selection_score(m)
            elapsed = time.time() - start_time

            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
                  f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"BDD DA mIoU: {m['bdd_da_miou']:.4f} | BDD Lane IoU: {m['bdd_ll_iou']:.4f} | "
                  f"BDD Lane Recall: {m['bdd_ll_recall']:.4f} | IDD DA mIoU: {m['idd_da_miou']:.4f}")

            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["epoch", "lr", "train_loss", "val_loss", *m.keys()])
                writer.writerow([epoch, scheduler.get_last_lr()[0], epoch_train_loss, val_loss, *m.values()])

            is_best = score > best_score
            if is_best:
                best_score = score
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': val_loss,
                'metrics': m,
                'best_score': best_score,
                'args': vars(args),
            }
            torch.save(checkpoint, latest_path)
            if is_best:
                torch.save(checkpoint, best_path)
                print(f"  --> Saved new best checkpoint: '{best_path}' (score {score:.4f})")

    except KeyboardInterrupt:
        print("\n[!] Training manually interrupted. Last completed epoch is in latest_model.pth.")

    print("=" * 65)
    print(f"Training Complete! Best score {best_score:.4f} -> '{best_path}'")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TwinDrive (dual-head drivable + lane segmentation)")
    parser.add_argument("--epochs", type=int, default=30, help="Total epochs for the run (cosine schedule length)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak LR (matches the base paper's 5e-4)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="twin/weights")
    parser.add_argument("--resume", type=str, default="", help="Path to latest_model.pth to continue a run")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Random subset size, for smoke tests")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Random subset size, for smoke tests")
    train(parser.parse_args())
