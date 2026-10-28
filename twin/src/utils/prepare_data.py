import os
import cv2
import numpy as np
from tqdm import tqdm

def process_bdd100k_drivable(raw_mask_path, target_width=640, target_height=360):
    mask = cv2.imread(raw_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
        
    # Fixed: The background is black (0), so anything greater than 0 is the drivable road
    drivable_binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    drivable_resized = cv2.resize(drivable_binary, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return drivable_resized

def process_idd_lite_drivable(raw_mask_path, target_width=640, target_height=360):
    mask = cv2.imread(raw_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # IDD-Lite level-1 ids: 0 = drivable (road, parking), 1 = non-drivable (sidewalk, rail track),
    # 2-6 = living things / vehicles / roadside objects / far objects / sky, 255 = void.
    # Only class 0 is road.
    drivable_binary = np.where(mask == 0, 255, 0).astype(np.uint8)
    drivable_resized = cv2.resize(drivable_binary, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return drivable_resized

def build_filename_map(search_dir, extensions=(".png", ".jpg", ".jpeg")):
    file_map = {}
    if not search_dir or not os.path.exists(search_dir):
        return file_map

    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.lower().endswith(extensions):
                full_path = os.path.join(root, f)
                stem = os.path.splitext(f)[0]
                
                # Strip all known dataset suffix variants, adding _lane_id
                clean_stem = stem
                for suffix in [
                    "_drivable_id", "_drivable_color", "_lane_id", "_leftImg8bit",
                    "_gtFine_labellevel3Ids", "_gtFine_labellevel2Ids", 
                    "_gtFine_labellevel1Ids", "_gtFine_labelIds", "_gtFine_polygons",
                    "_gtFine", "_image", "_label"
                ]:
                    clean_stem = clean_stem.replace(suffix, "")
                
                file_map[clean_stem] = full_path
                file_map[stem] = full_path
                
                prefix = stem.split("_")[0]
                if prefix not in file_map:
                    file_map[prefix] = full_path
                
    return file_map

def convert_split(img_dir, mask_dir, lane_dir, output_dir, split_name="train", dataset_type="bdd100k", fallback_mask_root=None):
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f" [Skipping] Split path missing: {img_dir} or {mask_dir}")
        return 0

    image_map = build_filename_map(img_dir)
    mask_map = build_filename_map(mask_dir)
    lane_map = build_filename_map(lane_dir) if lane_dir else {}
    if lane_dir and not lane_map:
        raise FileNotFoundError(f"Lane folder '{lane_dir}' was given but contains no lane masks. "
                                f"Check the path before preprocessing.")

    matched_stems = set(image_map.keys()).intersection(set(mask_map.keys()))
    
    if len(matched_stems) == 0 and fallback_mask_root and os.path.exists(fallback_mask_root):
        mask_map = build_filename_map(fallback_mask_root)
        matched_stems = set(image_map.keys()).intersection(set(mask_map.keys()))

    if len(matched_stems) == 0:
        print(f" [Warning] No matching stems found in {img_dir}")
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

        # 1. Process Drivable Mask
        if dataset_type.lower() == "bdd100k":
            drivable_mask = process_bdd100k_drivable(mask_path)
        else:
            drivable_mask = process_idd_lite_drivable(mask_path)

        if drivable_mask is None:
            continue

        # 2. Process Image
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_resized = cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR)

        # 3. Process Lane Mask (Real data for BDD100k, Zeros for IDD Lite)
        lane_mask = np.zeros((360, 640), dtype=np.uint8)
        if dataset_type.lower() == "bdd100k":
            if stem not in lane_map:
                raise KeyError(f"No lane mask found for BDD image '{stem}'")
            raw_lane = cv2.imread(lane_map[stem], cv2.IMREAD_GRAYSCALE)
            if raw_lane is None:
                raise IOError(f"Could not read lane mask '{lane_map[stem]}'")
            lane_binary = np.where(raw_lane > 0, 255, 0).astype(np.uint8)
            lane_mask = cv2.resize(lane_binary, (640, 360), interpolation=cv2.INTER_NEAREST)
        
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
    # Supports both datasets/100k (full) and data/raw/bdd100k_images_10k/10k (local 10k)
    if os.path.exists("datasets/100k"):
        bdd_root = "datasets"
        bdd_img_root = os.path.join(bdd_root, "100k")
    else:
        bdd_root = base_raw
        bdd_img_root = os.path.join(bdd_root, "bdd100k_images_10k", "10k")

    bdd_mask_root = os.path.join(bdd_root, "bdd_seg_gt")
    bdd_lane_root = os.path.join(bdd_root, "bdd_lane_gt")

    # Train Split
    convert_split(
        img_dir=os.path.join(bdd_img_root, "train"),
        mask_dir=os.path.join(bdd_mask_root, "train"),
        lane_dir=os.path.join(bdd_lane_root, "train"),
        output_dir=processed_root,
        split_name="train",
        dataset_type="bdd100k",
    )

    # Validation Split
    convert_split(
        img_dir=os.path.join(bdd_img_root, "val"),
        mask_dir=os.path.join(bdd_mask_root, "val"),
        lane_dir=os.path.join(bdd_lane_root, "val"),
        output_dir=processed_root,
        split_name="val",
        dataset_type="bdd100k",
    )

    # -------------------------------------------------------------
    # 2. IDD Lite Dataset Parsing
    # -------------------------------------------------------------
    print("\n[2/2] Scanning IDD Lite Directory Tree...")
    idd_img_root = os.path.join(base_raw, "idd20k_lite", "leftImg8bit")
    idd_mask_root = os.path.join(base_raw, "idd20k_lite", "gtFine")

    convert_split(
        img_dir=os.path.join(idd_img_root, "train"),
        mask_dir=os.path.join(idd_mask_root, "train"),
        lane_dir=None,  # No lane GT for IDD Lite
        output_dir=processed_root,
        split_name="train",
        dataset_type="idd_lite",
        fallback_mask_root=idd_mask_root
    )

    convert_split(
        img_dir=os.path.join(idd_img_root, "val"),
        mask_dir=os.path.join(idd_mask_root, "val"),
        lane_dir=None,  # No lane GT for IDD Lite
        output_dir=processed_root,
        split_name="val",
        dataset_type="idd_lite",
        fallback_mask_root=idd_mask_root
    )

    print("=" * 65)
    print("Dataset parsing and normalization complete!")
    print("=" * 65)
