import os, random, collections
import cv2

random.seed(0)
for split in ["train", "val"]:
    root = f"data/processed/{split}"
    names = os.listdir(os.path.join(root, "images"))
    print(split, collections.Counter(n.split("_")[0] for n in names))   # how many per dataset

    bdd = [n for n in names if n.startswith("bdd100k_")]
    sample = random.sample(bdd, 50)
    with_lanes = sum(cv2.imread(os.path.join(root, "masks_lanes", n), 0).any() for n in sample)
    print(f"  BDD lane masks with lane pixels: {with_lanes}/50")