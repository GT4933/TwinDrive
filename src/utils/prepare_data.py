import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

def process_bdd100k_drivable(raw_mask_path, target_width=640, target_height=360):
    """
    Converts raw BDD100K drivable area masks to binary masks.
    BDD100K drivable index annotations (inside labels/ folder):
    - 0: Background
    - 1: Main Drivable Lane
    - 2: Alternative Drivable Lane
    """
    mask = cv2.imread(raw_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
        
    # Combine main (1) and alternative (2) drivable areas into single binary mask (255)
    drivable_binary = np.where((mask == 1) | (mask == 2), 255, 0).astype(np.uint8)
    drivable_resized = cv2.resize(drivable_binary, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return drivable_resized

def process_idd_lite_drivable(raw_mask_path, target_width=640, target_height=360):
    """
    Converts raw IDD Lite segmentation masks to binary drivable space.
    IDD Lite annotations:
    - Class 0: Road (Paved)
    - Class 1: Drivable Fallback / Unpaved Road
    """
    mask = cv2.imread(raw_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
        
    # Map both paved road (0) and unpaved/fallback drivable surfaces (1)
    drivable_binary = np.where((mask == 0) | (mask == 1), 255, 0).astype(np.uint8)
    drivable_resized = cv2.resize(drivable_binary, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return drivable_resized

def build_filename_map(search_dir, extensions=(".png", ".jpg", ".jpeg")):
    """
    Recursively scans a directory and builds a dictionary mapping 
    clean file stem (without dataset suffixes) to its full file path.
    """
    file_map = {}
    if not os.path.exists(search_dir):
        return file_map

    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.lower().endswith(extensions):
                full_path = os.path.join(root, f)
                stem = os.path.splitext(f)[0]
                
                # Strip all known dataset suffix variants
                clean_stem = stem
                for suffix in [
                    "_drivable_id", "_drivable_color", "_leftImg8bit",
                    "_gtFine_labellevel3Ids", "_gtFine_labellevel2Ids", 
                    "_gtFine_labellevel1Ids", "_gtFine_labelIds", "_gtFine_polygons",
                    "_gtFine", "_image", "_label"
                ]:
                    clean_stem = clean_stem.replace(suffix, "")
                
                # Primary key: Cleaned stem
                file_map[clean_stem] = full_path
                # Fallback key: Original stem
                file_map[stem] = full_path
                # Fallback key: Prefix before first underscore (handles numeric frame IDs)
                prefix = stem.split("_")[0]
                if prefix not in file_map:
                    file_map[prefix] = full_path
                
    return file_map

def convert_split(img_dir, mask_dir, output_dir, split_name="train", dataset_type="bdd100k", fallback_mask_root=None):
    """
    Processes a specific split (train or val) for a given dataset source.
    """
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f" [Skipping] Split path missing: {img_dir} or {mask_dir}")
        return 0

    image_map = build_filename_map(img_dir)
    mask_map = build_filename_map(mask_dir)

    matched_stems = set(image_map.keys()).intersection(set(mask_map.keys()))
    
    # Fallback search across root mask directory if split mismatch occurs
    if len(matched_stems) == 0 and fallback_mask_root and os.path.exists(fallback_mask_root):
        print(f"  [Info] Retrying search across full mask root: {fallback_mask_root}")
        mask_map = build_filename_map(fallback_mask_root)
        matched_stems = set(image_map.keys()).intersection(set(mask_map.keys()))

    if len(matched_stems) == 0:
        print(f" [Warning] No matching stems found in {img_dir}")
        sample_imgs = list(image_map.keys())[:3]
        sample_masks = list(mask_map.keys())[:3]
        print(f"   --> Diagnostic Image Stems Found : {sample_imgs}")
        print(f"   --> Diagnostic Mask Stems Found  : {sample_masks}")
        return 0

    target_split_dir = os.path.join(output_dir, split_name)
    for sub in ["images", "masks_drivable", "masks_lanes"]:
        os.makedirs(os.path.join(target_split_dir, sub), exist_ok=True)

    count = 0
    desc = f"Parsing {dataset_type.upper()} [{split_name.upper()}]"
    for stem in tqdm(matched_stems, desc=desc):
        img_path = image_map[stem]
        mask_path = mask_map[stem]

        out_filename = f"{dataset_type}_{stem}.png"

        # Parse mask based on dataset source
        if dataset_type.lower() == "bdd100k":
            drivable_mask = process_bdd100k_drivable(mask_path)
        else:
            drivable_mask = process_idd_lite_drivable(mask_path)

        if drivable_mask is None:
            continue

        # Read and resize camera frame to (640, 360)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_resized = cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR)

        # Blank lane mask placeholder for unstructured scenes missing lane annotations
        lane_mask = np.zeros((360, 640), dtype=np.uint8)

        # Save standardized outputs
        cv2.imwrite(os.path.join(target_split_dir, "images", out_filename), img_resized)
        cv2.imwrite(os.path.join(target_split_dir, "masks_drivable", out_filename), drivable_mask)
        cv2.imwrite(os.path.join(target_split_dir, "masks_lanes", out_filename), lane_mask)
        count += 1

    print(f"  --> Processed {count} samples into '{target_split_dir}'")
    return count

if __name__ == "__main__":
    print("=" * 65)
    print("        RAW DATASET INGESTION & CONVERSION PIPELINE")
    print("=" * 65)
    
    base_raw = "data/raw"
    processed_root = "data/processed"

    # -------------------------------------------------------------
    # 1. BDD100K Dataset Parsing
    # -------------------------------------------------------------
    print("\n[1/2] Scanning BDD100K Directory Tree...")
    bdd_img_root = os.path.join(base_raw, "bdd100k_images_10k", "10k")
    bdd_mask_root = os.path.join(base_raw, "bdd100k_drivable_maps", "labels")

    # Handle train split
    convert_split(
        img_dir=os.path.join(bdd_img_root, "train"),
        mask_dir=os.path.join(bdd_mask_root, "train"),
        output_dir=processed_root,
        split_name="train",
        dataset_type="bdd100k",
        fallback_mask_root=bdd_mask_root
    )

    # Handle validation split
    convert_split(
        img_dir=os.path.join(bdd_img_root, "val"),
        mask_dir=os.path.join(bdd_mask_root, "val"),
        output_dir=processed_root,
        split_name="val",
        dataset_type="bdd100k",
        fallback_mask_root=bdd_mask_root
    )

    # -------------------------------------------------------------
    # 2. IDD Lite Dataset Parsing
    # -------------------------------------------------------------
    print("\n[2/2] Scanning IDD Lite Directory Tree...")
    idd_img_root = os.path.join(base_raw, "idd20k_lite", "leftImg8bit")
    idd_mask_root = os.path.join(base_raw, "idd20k_lite", "gtFine")

    # Handle train split
    convert_split(
        img_dir=os.path.join(idd_img_root, "train"),
        mask_dir=os.path.join(idd_mask_root, "train"),
        output_dir=processed_root,
        split_name="train",
        dataset_type="idd_lite",
        fallback_mask_root=idd_mask_root
    )

    # Handle validation split
    convert_split(
        img_dir=os.path.join(idd_img_root, "val"),
        mask_dir=os.path.join(idd_mask_root, "val"),
        output_dir=processed_root,
        split_name="val",
        dataset_type="idd_lite",
        fallback_mask_root=idd_mask_root
    )

    print("=" * 65)
    print("Dataset parsing and normalization complete!")
    print("=" * 65)