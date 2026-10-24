"""
Adds data/processed/{train,val}/masks_ego/: the car's own lane ("direct" drivable area) as a 0/255 mask.

Source: BDD100K's official drivable maps (data/raw/bdd100k_drivable_maps/labels/<split>/<stem>_drivable_id.png),
where 0 = background, 1 = direct (own lane), 2 = alternative (other lanes). Checked on 400 val maps:
value 1 is centred (centroid x 0.49, std 0.11) and owns the bottom-centre of the frame; value 2 is spread
left/right (std 0.24) and never does; together they cover 16.5% of pixels, matching bdd_seg_gt's drivable area.

Note: datasets/bdd_seg_gt (used for the drivable mask) does NOT carry this split: its value 127 covers ~97% of
all drivable pixels and 191 is a thin sliver, so it cannot be used for the ego lane.

IDD-Lite has no direct/alternative split, so its ego masks are written as zeros and the loss ignores them
(same domain gate as lanes).

Run from the repo root after prepare_data.py:  python twin/src/utils/make_ego_masks.py
"""
import os
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

BDD_DIRECT_VALUE = 1
PROCESSED_ROOT = "data/processed"
BDD_DRIVABLE_MAPS = "data/raw/bdd100k_drivable_maps/labels"


def _convert(job):
    out_path, raw_path = job
    if raw_path is None:
        ego = np.zeros((360, 640), dtype=np.uint8)
    else:
        mask = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Could not read drivable map '{raw_path}'")
        ego = np.where(mask == BDD_DIRECT_VALUE, 255, 0).astype(np.uint8)
        ego = cv2.resize(ego, (640, 360), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_path, ego)


def build_jobs(split):
    img_dir = os.path.join(PROCESSED_ROOT, split, "images")
    out_dir = os.path.join(PROCESSED_ROOT, split, "masks_ego")
    os.makedirs(out_dir, exist_ok=True)
    jobs = []
    for name in sorted(os.listdir(img_dir)):
        if name.startswith("bdd100k_"):
            stem = os.path.splitext(name[len("bdd100k_"):])[0]
            raw = os.path.join(BDD_DRIVABLE_MAPS, split, stem + "_drivable_id.png")
            if not os.path.exists(raw):
                raise FileNotFoundError(f"No BDD drivable map for '{name}' at '{raw}'")
        else:
            raw = None
        jobs.append((os.path.join(out_dir, name), raw))
    return jobs


if __name__ == "__main__":
    for split in ["train", "val"]:
        jobs = build_jobs(split)
        with Pool(4) as pool:
            list(tqdm(pool.imap_unordered(_convert, jobs, chunksize=64), total=len(jobs), desc=f"ego masks [{split}]"))
        print(f"  --> wrote {len(jobs)} masks to {PROCESSED_ROOT}/{split}/masks_ego")
