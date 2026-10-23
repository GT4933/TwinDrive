import os
import sys
import argparse

# Project root (twin/) for src.*, and base_paper/ for the TwinLiteNet baseline
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "twin"))
sys.path.append(os.path.join(ROOT, "base_paper"))

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.dataset import TwinLiteNetDataset
from src.utils.metrics import BinarySegMetric


def build_model(arch, weights, device):
    """Returns (model, transform, to_binary) so both architectures run through the same metric code."""
    if arch in ("twin", "v2"):
        if arch == "twin":
            from src.models.twinlitenet import TwinLiteNet
            model = TwinLiteNet()
        else:
            from src.models.twindrive_v2 import TwinDriveV2
            model = TwinDriveV2()
        # TwinDrive trains on ImageNet-normalised RGB and outputs one logit per head
        transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'drivable': 'mask', 'lanes': 'mask', 'ego': 'mask'})
        to_binary = lambda logits: torch.sigmoid(logits[:, 0].float()) > 0.5
    else:
        from model.TwinLite import TwinLiteNet as BaseTwinLiteNet
        model = BaseTwinLiteNet()
        # Base trains on RGB scaled to 0-1 (no mean/std) and outputs two-class logits
        transform = A.Compose([
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
            ToTensorV2()
        ], additional_targets={'drivable': 'mask', 'lanes': 'mask', 'ego': 'mask'})
        to_binary = lambda logits: logits.argmax(dim=1) == 1

    checkpoint = torch.load(weights, map_location=device)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    state = {k.replace("module.", "", 1): v for k, v in state.items()}  # strip DataParallel prefix
    model.load_state_dict(state)
    return model.to(device).eval(), transform, to_binary


def evaluate(arch, weights, data_dir="data/processed/val", batch_size=8, workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, to_binary = build_model(arch, weights, device)
    with_ego = arch == "v2"
    dataset = TwinLiteNetDataset(data_dir=data_dir, is_train=False, transform=transform, with_ego=with_ego)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    da_bdd, da_idd, ll_bdd, ego_bdd = BinarySegMetric(), BinarySegMetric(), BinarySegMetric(), BinarySegMetric()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {arch}"):
            images, drivable_masks, lane_masks, lane_valids = batch[:4]
            outputs = model(images.to(device))
            d_pred, l_pred = to_binary(outputs[0]).cpu(), to_binary(outputs[1]).cpu()
            is_bdd = lane_valids > 0.5
            if is_bdd.any():
                da_bdd.update(d_pred[is_bdd], drivable_masks[is_bdd])
                ll_bdd.update(l_pred[is_bdd], lane_masks[is_bdd])
                if with_ego:
                    ego_bdd.update(to_binary(outputs[2]).cpu()[is_bdd], batch[4][is_bdd])
            if (~is_bdd).any():
                da_idd.update(d_pred[~is_bdd], drivable_masks[~is_bdd])

    print("=" * 65)
    print(f"  {arch} | {weights} | {len(dataset)} images from {data_dir}")
    print("-" * 65)
    if da_bdd.num_pixels():
        print(f"  BDD  drivable : mIoU {da_bdd.miou():.4f} | IoU {da_bdd.iou():.4f} | acc {da_bdd.pixel_accuracy():.4f}")
        print(f"  BDD  lanes    : IoU  {ll_bdd.iou():.4f} | recall (paper 'acc') {ll_bdd.recall():.4f}")
    if ego_bdd.num_pixels():
        print(f"  BDD  ego lane : mIoU {ego_bdd.miou():.4f} | IoU {ego_bdd.iou():.4f}")
    if da_idd.num_pixels():
        print(f"  IDD  drivable : mIoU {da_idd.miou():.4f} | IoU {da_idd.iou():.4f} | acc {da_idd.pixel_accuracy():.4f}")
    print("  (dataset-level confusion matrices; lanes scored on BDD only)")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score TwinDrive or the TwinLiteNet baseline with identical metrics")
    parser.add_argument("--arch", choices=["twin", "v2", "base"], required=True,
                        help="twin = TwinDrive v1, v2 = TwinDrive v2, base = TwinLiteNet baseline")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/processed/val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    evaluate(args.arch, args.weights, args.data_dir, args.batch_size, args.workers)
