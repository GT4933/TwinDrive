"""
IDD-Lite has no lane labels. This writes pseudo lane labels for the IDD training images from a trained teacher
model, keeping only pixels the teacher is confident about:

    255 = lane        (teacher probability > --hi)
      0 = not lane    (teacher probability < --lo)
    128 = ignore      (everything in between; excluded from the loss)

Output: data/processed/train/masks_lanes_pseudo/<same filename>. Train with `train.py --pseudo-lanes`.
There is no IDD lane ground truth, so pseudo-label quality cannot be measured, only inspected visually.

Run from the repo root:
    python twin/src/utils/make_pseudo_lanes.py --teacher twin/weights_v2_ego/best_model.pth
"""
import os
import sys
import argparse

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.models.twinlitenet import TwinLiteNet
from src.models.twindrive_v2 import TwinDriveV2, TwinDriveV2LaneFix

PSEUDO_LANE, PSEUDO_IGNORE = 255, 128
ARCHS = {"v1": TwinLiteNet, "v2": TwinDriveV2, "v2lf": TwinDriveV2LaneFix}


def load_teacher(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = ARCHS[checkpoint.get("args", {}).get("arch", "v1")]()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_teacher(args.teacher, device)
    img_dir = os.path.join(args.split_dir, "images")
    out_dir = os.path.join(args.split_dir, "masks_lanes_pseudo")
    os.makedirs(out_dir, exist_ok=True)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    names = sorted(f for f in os.listdir(img_dir) if f.startswith("idd_lite_"))
    lane_frac, ignore_frac = [], []

    with torch.no_grad():
        for name in tqdm(names, desc="pseudo lanes"):
            # Same preprocessing as training/inference: RGB, ImageNet normalisation
            rgb = cv2.cvtColor(cv2.imread(os.path.join(img_dir, name)), cv2.COLOR_BGR2RGB)
            x = torch.from_numpy((rgb.astype(np.float32) / 255.0 - mean) / std).permute(2, 0, 1)[None].to(device)
            prob = torch.sigmoid(teacher(x)[1])[0, 0].float().cpu().numpy()

            label = np.full(prob.shape, PSEUDO_IGNORE, dtype=np.uint8)
            label[prob > args.hi] = PSEUDO_LANE
            label[prob < args.lo] = 0
            cv2.imwrite(os.path.join(out_dir, name), label)
            lane_frac.append((label == PSEUDO_LANE).mean())
            ignore_frac.append((label == PSEUDO_IGNORE).mean())

    print(f"  --> wrote {len(names)} pseudo labels to '{out_dir}'")
    print(f"  mean lane pixels {np.mean(lane_frac):.4f} | mean ignored pixels {np.mean(ignore_frac):.4f} | "
          f"images with any lane {np.mean(np.array(lane_frac) > 0):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teacher-generated lane pseudo labels for IDD-Lite")
    parser.add_argument("--teacher", required=True, help="Checkpoint of a model trained with BDD lane labels")
    parser.add_argument("--split-dir", default="data/processed/train")
    parser.add_argument("--hi", type=float, default=0.9, help="Probability above which a pixel is labelled lane")
    parser.add_argument("--lo", type=float, default=0.1, help="Probability below which a pixel is labelled not-lane")
    main(parser.parse_args())
