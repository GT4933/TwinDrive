"""
Adds data/processed/{train,val}/masks_ego/: the car's own lane ("direct" drivable area) as a 0/255 mask.

bdd_seg_gt uses 0 = background, 127 = direct (ego lane), 191 = alternative (other lanes). We checked the
assignment on 300 val masks: 127 fills the bottom-centre of the frame in 75 of them, 191 in none, and
127 covers ~16.6% of pixels vs ~0.5% for 191. IDD-Lite has no direct/alternative split, so its ego
masks are written as zeros and the loss ignores them (same domain gate as lanes).

Run from the repo root after prepare_data.py:  python twin/src/utils/make_ego_masks.py
"""
import os
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

BDD_DIRECT_VALUE = 127
PROCESSED_ROOT = "data/processed"
BDD_SEG_ROOT = "datasets/bdd_seg_gt"


def _convert(job):
    out_path, raw_path = job
    if raw_path is None:
        ego = np.zeros((360, 640), dtype=np.uint8)
    else:
        mask = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Could not read drivable mask '{raw_path}'")
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
            raw = os.path.join(BDD_SEG_ROOT, split, stem + ".png")
            if not os.path.exists(raw):
                raise FileNotFoundError(f"No BDD drivable mask for '{name}' at '{raw}'")
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
