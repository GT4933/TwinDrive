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

from src.models.twindrive_v1 import TwinDriveV1
from src.models.twindrive_v2 import TwinDriveV2, TwinDriveV2LaneFix
from src.utils.dataset import TwinLiteNetDataset
from src.utils.metrics import BinarySegMetric

ARCHS = {"v1": TwinDriveV1, "v2": TwinDriveV2, "v2lf": TwinDriveV2LaneFix}

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

class TverskyLoss(nn.Module):
    """
    Per-image soft Tversky loss: 1 - TP / (TP + alpha*FP + beta*FN).
    alpha > beta penalises false positives more (raises precision); alpha = beta = 0.5 is Dice.
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, targets, reduce=True):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        tp = (probs * targets).sum(dim=(1, 2))
        fp = (probs * (1 - targets)).sum(dim=(1, 2))
        fn = ((1 - probs) * targets).sum(dim=(1, 2))
        loss = 1. - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return loss.mean() if reduce else loss

class TwinLiteLoss(nn.Module):
    """
    Multi-Task Loss with Domain-Gated Masking for Unlabeled Lanes.
    Returns a dict of per-task losses; TaskWeighting combines them into one number.
    """
    def __init__(self, lane_loss="dice", tversky_alpha=0.7, tversky_beta=0.3, lane_overlap_weight=1.5,
                 lane_focal_gamma=0.0):
        super(TwinLiteLoss, self).__init__()
        # reduction='none' allows us to isolate loss per image in the batch
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dice = DiceLoss()
        self.lane_overlap = TverskyLoss(tversky_alpha, tversky_beta) if lane_loss == "tversky" else self.dice
        self.lane_overlap_weight = lane_overlap_weight
        # > 0 turns the lane BCE into focal BCE: easy (mostly background) pixels are down-weighted by (1 - p_t)^gamma
        self.lane_focal_gamma = lane_focal_gamma

    def _gated(self, logits, targets, valids, overlap, overlap_weight, focal_gamma=0.0):
        """
        Per-image BCE (or focal BCE) + overlap loss, averaged over the images whose labels exist (weighted by
        valids). Pixels labelled TwinLiteNetDataset.LANE_IGNORE (uncertain pseudo labels) are left out of both terms.
        """
        keep = (targets != TwinLiteNetDataset.LANE_IGNORE).float()
        targets = targets.clamp(max=1).float() * keep
        pixel_bce = self.bce(logits, targets)
        if focal_gamma > 0:
            pixel_bce = (1 - torch.exp(-pixel_bce)) ** focal_gamma * pixel_bce   # exp(-BCE) = p_t
        bce = (pixel_bce * keep).sum(dim=(1, 2)) / keep.sum(dim=(1, 2)).clamp(min=1.0)
        # Ignored pixels get a large negative logit, i.e. probability 0, so they add nothing to the overlap term
        overlap_logits = logits.masked_fill(keep == 0, -1e4)
        per_image = bce + overlap_weight * overlap(overlap_logits, targets, reduce=False)
        if valids is None:
            return per_image.mean()
        # Prevent division by zero if entire batch happens to be IDD
        return (per_image * valids).sum() / torch.clamp(valids.sum(), min=1.0)

    def forward(self, pred_drivable, pred_lanes, target_drivable, target_lanes, lane_valids=None,
                pred_ego=None, target_ego=None):
        # Heads output (B, 1, H, W) logits; drop the channel dim to match (B, H, W) targets
        drivable_logits = pred_drivable.squeeze(1)

        # ---------------------------------------------------
        # Drivable Area Loss (Supervised across ALL frames)
        # ---------------------------------------------------
        loss_drivable_bce = self.bce(drivable_logits, target_drivable.float()).mean()
        loss_drivable_dice = self.dice(drivable_logits, target_drivable, reduce=True)
        losses = {"drivable": loss_drivable_bce + loss_drivable_dice}

        # ---------------------------------------------------
        # Lane Line Loss (Domain-Gated via Selective Masking)
        # ---------------------------------------------------
        losses["lane"] = self._gated(pred_lanes.squeeze(1), target_lanes, lane_valids, self.lane_overlap,
                                     self.lane_overlap_weight, self.lane_focal_gamma)

        # Ego lane (BDD "direct" area): BDD frames only, since IDD has no own-lane labels
        # (IDD frames may carry a fractional lane weight for pseudo labels, so gate on the domain explicitly)
        if pred_ego is not None:
            bdd_only = (lane_valids > 0.5).float() if lane_valids is not None else None
            losses["ego"] = self._gated(pred_ego.squeeze(1), target_ego, bdd_only, self.dice, 1.0)
        return losses

class TaskWeighting(nn.Module):
    """
    'fixed': plain sum of task losses (the original recipe).
    'uncertainty': Kendall et al. (2018) learned weighting, total = sum(exp(-s_t) * L_t + s_t), where s_t is a
    learned log-variance per task; tasks that stay noisy get down-weighted automatically.
    """
    def __init__(self, tasks, mode="fixed"):
        super(TaskWeighting, self).__init__()
        self.mode = mode
        self.tasks = list(tasks)
        if mode == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(len(self.tasks)))

    def forward(self, losses):
        if self.mode == "fixed":
            return sum(losses[t] for t in self.tasks)
        return sum(torch.exp(-self.log_vars[i]) * losses[t] + self.log_vars[i] for i, t in enumerate(self.tasks))

    def weights(self):
        if self.mode == "fixed":
            return {t: 1.0 for t in self.tasks}
        return {t: float(torch.exp(-self.log_vars[i])) for i, t in enumerate(self.tasks)}

def unpack(batch, device):
    """(images, drivable, lanes, lane_valids, ego-or-None) on device; ego exists only for with_ego datasets."""
    images, drivable, lanes, valids = (t.to(device, non_blocking=True) for t in batch[:4])
    ego = batch[4].to(device, non_blocking=True) if len(batch) > 4 else None
    return images, drivable, lanes, valids, ego

def compute_losses(model, criterion, images, drivable, lanes, valids, ego):
    outputs = model(images)
    pred_ego = outputs[2] if len(outputs) > 2 else None
    return outputs, criterion(outputs[0], outputs[1], drivable, lanes, valids, pred_ego, ego)

# ---------------------------------------------------------
# 2. Validation Pass (dataset-level metrics, reported per domain)
# ---------------------------------------------------------
def validate(model, val_loader, criterion, weighting, device, threshold=0.5):
    """
    Returns (mean val loss, metrics dict). Confusion matrices are accumulated over the whole
    val set and split by domain: lanes (and ego lane) are scored on BDD frames only.
    """
    model.eval()
    val_loss = 0.0
    da_bdd, da_idd, ll_bdd, ego_bdd = BinarySegMetric(), BinarySegMetric(), BinarySegMetric(), BinarySegMetric()

    with torch.no_grad():
        for batch in val_loader:
            images, drivable_masks, lane_masks, lane_valids, ego_masks = unpack(batch, device)

            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                outputs, losses = compute_losses(model, criterion, images, drivable_masks, lane_masks, lane_valids, ego_masks)
                loss = weighting(losses)
            val_loss += loss.item()

            d_pred = torch.sigmoid(outputs[0].squeeze(1).float()) > threshold
            l_pred = torch.sigmoid(outputs[1].squeeze(1).float()) > threshold

            is_bdd = lane_valids > 0.5
            if is_bdd.any():
                da_bdd.update(d_pred[is_bdd], drivable_masks[is_bdd])
                ll_bdd.update(l_pred[is_bdd], lane_masks[is_bdd])
                if ego_masks is not None:
                    e_pred = torch.sigmoid(outputs[2].squeeze(1).float()) > threshold
                    ego_bdd.update(e_pred[is_bdd], ego_masks[is_bdd])
            if (~is_bdd).any():
                da_idd.update(d_pred[~is_bdd], drivable_masks[~is_bdd])

    nan = float("nan")
    metrics = {
        "bdd_da_miou": da_bdd.miou() if da_bdd.num_pixels() else nan,
        "bdd_ll_iou": ll_bdd.iou() if ll_bdd.num_pixels() else nan,
        "bdd_ll_recall": ll_bdd.recall() if ll_bdd.num_pixels() else nan,
        "idd_da_miou": da_idd.miou() if da_idd.num_pixels() else nan,
    }
    if ego_bdd.num_pixels():
        metrics["bdd_ego_miou"] = ego_bdd.miou()
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
    with_ego = args.arch in ("v2", "v2lf")
    print("=" * 65)
    print("           TWINDRIVE MULTI-TASK TRAINING ENGINE")
    print("=" * 65)
    print(f"Target Compute Hardware : {device}")
    if torch.cuda.is_available():
        print(f"GPU Name                 : {torch.cuda.get_device_name(0)}")
    print(f"Architecture             : {args.arch} | lane loss {args.lane_loss} | weighting {args.task_weighting} "
          f"| geometric aug {args.geometric_aug} | IDD pseudo lanes {args.pseudo_lanes}")

    os.makedirs(args.out_dir, exist_ok=True)

    train_dataset = subset(TwinLiteNetDataset(data_dir="data/processed/train", is_train=True, with_ego=with_ego,
                                              geometric_aug=args.geometric_aug, pseudo_lanes=args.pseudo_lanes,
                                              pseudo_weight=args.pseudo_weight, pseudo_dir_name=args.pseudo_dir),
                           args.max_train_samples)
    val_dataset = subset(TwinLiteNetDataset(data_dir="data/processed/val", is_train=False, with_ego=with_ego),
                         args.max_val_samples)
    print(f"Dataset Allocation       : {len(train_dataset)} Train | {len(val_dataset)} Val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    # Val workers are re-spawned each epoch and kept few: on Windows, persistent train workers plus a second
    # full set of val workers exhausted memory and hung the run
    val_workers = min(args.workers, 2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=val_workers, pin_memory=True)

    model = ARCHS[args.arch]().to(device)
    print(f"Model Parameters         : {sum(p.numel() for p in model.parameters()):,}")
    if args.init_weights:
        # Fine-tuning: start from trained weights but a fresh optimizer, schedule and log
        init = torch.load(args.init_weights, map_location=device)
        if init.get("args", {}).get("arch", "v1") != args.arch:
            raise ValueError(f"--init-weights '{args.init_weights}' is not a --arch {args.arch} checkpoint")
        model.load_state_dict(init["model_state_dict"])
        print(f"Initialised weights from : '{args.init_weights}'")
    criterion = TwinLiteLoss(lane_loss=args.lane_loss, tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta,
                             lane_overlap_weight=args.lane_overlap_weight, lane_focal_gamma=args.lane_focal_gamma)
    tasks = ["drivable", "lane"] + (["ego"] if with_ego else [])
    weighting = TaskWeighting(tasks, mode=args.task_weighting).to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(weighting.parameters()), lr=args.lr, weight_decay=1e-4)
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
        for key in ("epochs", "arch"):
            saved = checkpoint['args'].get(key, "v1" if key == "arch" else None)
            if saved != getattr(args, key):
                raise ValueError(f"Checkpoint was trained with --{key} {saved}; resuming with --{key} "
                                 f"{getattr(args, key)} would not continue the same run. Pass --{key} {saved}.")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'weighting_state_dict' in checkpoint:
            weighting.load_state_dict(checkpoint['weighting_state_dict'])
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

            for batch in pbar:
                images, drivable_masks, lane_masks, lane_valids, ego_masks = unpack(batch, device)

                optimizer.zero_grad()

                # Cast forward pass to mixed precision
                with torch.amp.autocast(device.type, enabled=use_amp):
                    _, losses = compute_losses(model, criterion, images, drivable_masks, lane_masks, lane_valids, ego_masks)
                    loss = weighting(losses)

                # Scaled backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

            scheduler.step()

            epoch_train_loss = running_loss / len(train_loader)
            val_loss, m = validate(model, val_loader, criterion, weighting, device)
            score = selection_score(m)
            elapsed = time.time() - start_time

            ego_str = f" | BDD Ego mIoU: {m['bdd_ego_miou']:.4f}" if "bdd_ego_miou" in m else ""
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
                  f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"BDD DA mIoU: {m['bdd_da_miou']:.4f} | BDD Lane IoU: {m['bdd_ll_iou']:.4f} | "
                  f"BDD Lane Recall: {m['bdd_ll_recall']:.4f} | IDD DA mIoU: {m['idd_da_miou']:.4f}{ego_str}")
            if args.task_weighting == "uncertainty":
                print("  task weights: " + ", ".join(f"{t} {w:.3f}" for t, w in weighting.weights().items()))

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
                'weighting_state_dict': weighting.state_dict(),
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
    parser.add_argument("--arch", choices=sorted(ARCHS), default="v1",
                        help="v1 = original TwinDrive; v2 = skip decoders, stride 8, HFF, coordinate attention, ego head")
    parser.add_argument("--epochs", type=int, default=30, help="Total epochs for the run (cosine schedule length)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak LR (matches the base paper's 5e-4)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="twin/weights")
    parser.add_argument("--resume", type=str, default="", help="Path to latest_model.pth to continue a run")
    parser.add_argument("--lane-loss", choices=["dice", "tversky"], default="dice",
                        help="Overlap term of the lane loss (tversky uses alpha=0.7 on FP, beta=0.3 on FN)")
    parser.add_argument("--tversky-alpha", type=float, default=0.7, help="Tversky weight on false positives")
    parser.add_argument("--tversky-beta", type=float, default=0.3, help="Tversky weight on false negatives")
    parser.add_argument("--lane-overlap-weight", type=float, default=1.5, help="Weight of the lane Dice/Tversky term")
    parser.add_argument("--lane-focal-gamma", type=float, default=0.0,
                        help="> 0 uses focal BCE for lanes (base paper uses 2); 0 = plain BCE")
    parser.add_argument("--task-weighting", choices=["fixed", "uncertainty"], default="fixed")
    parser.add_argument("--geometric-aug", action="store_true", help="Random scale/shift/rotate augmentation")
    parser.add_argument("--pseudo-lanes", action="store_true",
                        help="Supervise IDD lanes with teacher pseudo labels (run utils/make_pseudo_lanes.py first)")
    parser.add_argument("--pseudo-weight", type=float, default=0.5, help="Loss weight of pseudo-labelled IDD frames")
    parser.add_argument("--pseudo-dir", default="masks_lanes_pseudo",
                        help="Pseudo-label folder: masks_lanes_pseudo (self-training) or masks_lanes_pseudo_yolop")
    parser.add_argument("--init-weights", type=str, default="", help="Start from a trained checkpoint (fine-tuning)")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Random subset size, for smoke tests")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Random subset size, for smoke tests")
    train(parser.parse_args())
