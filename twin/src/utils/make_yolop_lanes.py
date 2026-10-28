"""
IDD lane pseudo labels from an *external* teacher, ported from GT4933's `g` branch
(generate_idd_lane_labels.py + clean_idd_labels.py):

  1. YOLOP (hustvl, trained on BDD100K) lane-line segmentation, kept inside a lightly eroded drivable mask
  2. OpenCV paint detection: white/yellow pixels that are locally brighter than the asphalt (top-hat),
     kept inside a strongly eroded drivable mask so dirt shoulders are excluded
  3. Union of both, then shape filters: ribbon-like (solidity), perspective-aware thickness, elongation,
     minimum length, no horizontal stop lines
  4. Sanity rejection: masks that are too dense (> 6000 px) or too fragmented (> 8 pieces) are discarded

Differences from `g`, so labels plug into this branch's training without touching evaluation:
  - writes data/processed/train/masks_lanes_pseudo_yolop/ (train split only; validation keeps real BDD labels)
  - pixel values follow make_pseudo_lanes.py: 255 = lane, 0 = not lane, 128 = ignore. Images whose mask
     is empty or rejected are written as all-ignore, so they give no lane supervision (as `g` gated them
     with lane_valid = 0)
  - saves visual green-line overlays to visualizations_lanes/ (toggle with --no-vis)
  - `--no-yolop` runs the OpenCV half alone (no download needed)

Train with `train.py --pseudo-lanes --pseudo-dir masks_lanes_pseudo_yolop`.
Run from the repo root:  python twin/src/utils/make_yolop_lanes.py
"""
import os
import argparse

import cv2
import numpy as np
from tqdm import tqdm

PSEUDO_LANE, PSEUDO_IGNORE = 255, 128


def load_yolop(yolop_dir=None):
    """
    yolop_dir: a local copy of github.com/hustvl/YOLOP holding hubconf.py, lib/ and weights/End-to-end.pth
    (much smaller than torch.hub's full-repo zip, which includes demo videos). None = download via torch.hub.
    """
    import torch
    if yolop_dir:
        print(f"Loading pre-trained YOLOP teacher from local copy '{yolop_dir}'...")
        model = torch.hub.load(yolop_dir, "yolop", source="local", pretrained=True)
    else:
        print("Loading pre-trained YOLOP teacher from PyTorch Hub (hustvl/yolop)...")
        model = torch.hub.load("hustvl/yolop", "yolop", pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), device


def yolop_lanes(img_rgb, model, device):
    """YOLOP lane-line mask (0/255) at the image's own size."""
    import torch
    import torchvision.transforms as transforms
    h, w = img_rgb.shape[:2]
    tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
        transforms.ToTensor()(cv2.resize(img_rgb, (640, 640)))).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, ll_seg_out = model(tensor)
    mask = (torch.argmax(ll_seg_out, dim=1).squeeze().cpu().numpy() * 255).astype(np.uint8)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def paint_lanes(img_bgr):
    """Painted-marking candidates: white/yellow AND locally brighter than the surrounding road."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Top-hat keeps thin bright ridges (<= 19 px wide); wide uniform surfaces cancel out
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5)))
    _, local_contrast = cv2.threshold(tophat, 25, 255, cv2.THRESH_BINARY)
    # High-lightness HLS bands reject brown soil
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, np.array([0, 195, 0]), np.array([180, 255, 70]))
    yellow = cv2.inRange(hls, np.array([15, 135, 110]), np.array([32, 255, 255]))
    mask = cv2.bitwise_and(cv2.bitwise_or(white, yellow), local_contrast)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))


def shape_filter(raw):
    """Keeps components shaped like lane segments (thin, long, ribbon-like, not horizontal)."""
    h = raw.shape[0]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    out = np.zeros_like(raw)
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 35:                                               # noise specks
            continue
        ys, xs = np.where(labels == i)
        (_, cy), (bw, bh), _ = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.int32))
        thickness, length = min(bw, bh), max(bw, bh)
        if area / max(bw * bh, 1.0) < 0.32:                         # sprawling patches, not ribbons
            continue
        max_thickness = max(6.0, 22.0 * cy / float(h))              # lanes thin towards the horizon
        if thickness > max_thickness or length < 28 or length / max(thickness, 1.0) < 3.0:
            continue
        if stats[i, cv2.CC_STAT_WIDTH] > 3.0 * stats[i, cv2.CC_STAT_HEIGHT]:   # stop lines, speed bumps
            continue
        out[labels == i] = 255
    return out


def label_image(img_bgr, drivable, model=None, device=None):
    """Returns (pseudo label in 255/0/128, status) for one IDD image."""
    yolop_region = cv2.erode(drivable, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    paint_region = cv2.erode(drivable, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 15)))

    raw = cv2.bitwise_and(paint_lanes(img_bgr), paint_region)
    if model is not None:
        yolop = yolop_lanes(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), model, device)
        raw = cv2.bitwise_or(raw, cv2.bitwise_and(yolop, yolop_region))
    lanes = shape_filter(raw)

    ignore_all = np.full(lanes.shape, PSEUDO_IGNORE, dtype=np.uint8)
    pixels = int(np.count_nonzero(lanes))
    if pixels <= 120:
        return ignore_all, "empty"
    if pixels > 6000 or cv2.connectedComponents(lanes, connectivity=8)[0] - 1 > 8:
        return ignore_all, "rejected"
    return np.where(lanes > 0, PSEUDO_LANE, 0).astype(np.uint8), "labelled"


def main(args):
    split = args.split_dir
    img_dir, drv_dir = os.path.join(split, "images"), os.path.join(split, "masks_drivable")
    out_dir = os.path.join(split, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    vis_dir = os.path.join(split, args.vis_name) if not args.no_vis else None
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    model, device = (None, None) if args.no_yolop else load_yolop(args.yolop_dir)

    counts = {"labelled": 0, "empty": 0, "rejected": 0}
    saved_vis = 0
    for name in tqdm(sorted(f for f in os.listdir(img_dir) if f.startswith("idd_lite_")), desc="IDD lane labels"):
        img = cv2.imread(os.path.join(img_dir, name))
        drivable = cv2.imread(os.path.join(drv_dir, name), cv2.IMREAD_GRAYSCALE)
        if img is None or drivable is None:
            raise IOError(f"Could not read '{name}' or its drivable mask")
        label, status = label_image(img, drivable, model, device)
        cv2.imwrite(os.path.join(out_dir, name), label)
        counts[status] += 1

        if vis_dir:
            vis_path = os.path.join(vis_dir, name)
            if status == "labelled":
                overlay = img.copy()
                overlay[label == PSEUDO_LANE] = [0, 255, 0]  # Green lane lines
                cv2.imwrite(vis_path, overlay)
                saved_vis += 1
            elif os.path.exists(vis_path):
                os.remove(vis_path)

    total = sum(counts.values())
    print(f"  --> {counts['labelled']}/{total} IDD images labelled, {counts['empty']} empty, "
          f"{counts['rejected']} rejected (dense/fragmented); unlabelled ones are written as all-ignore")
    if vis_dir:
        print(f"  --> Saved {saved_vis} active lane visualizations to '{vis_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDD lane pseudo labels from YOLOP + OpenCV paint detection")
    parser.add_argument("--split-dir", default="data/processed/train")
    parser.add_argument("--no-yolop", action="store_true", help="OpenCV paint detection only (no download)")
    parser.add_argument("--yolop-dir", default=None,
                        help="Local YOLOP copy (hubconf.py, lib/, weights/End-to-end.pth); default downloads via torch.hub")
    parser.add_argument("--out-name", default="masks_lanes_pseudo_yolop",
                        help="Output folder inside --split-dir; train with --pseudo-dir <same name>")
    parser.add_argument("--vis-name", default="visualizations_lanes",
                        help="Folder inside --split-dir to save green overlay visualizations")
    parser.add_argument("--no-vis", action="store_true", help="Skip saving visual overlay frames")
    main(parser.parse_args())
