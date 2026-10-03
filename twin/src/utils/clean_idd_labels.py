import os
import cv2
import numpy as np
from tqdm import tqdm

def clean_dataset(data_split="data/processed/train"):
    lanes_dir = os.path.join(data_split, "masks_lanes")
    vis_dir = os.path.join(data_split, "visualizations_lanes")
    
    if not os.path.exists(lanes_dir): return
        
    # ONLY load files that belong to the IDD dataset
    files = [f for f in os.listdir(lanes_dir) if f.endswith(".png") and "idd_lite" in f]
    print(f"\nRunning Automated Sanity Check on {len(files)} IDD masks in {data_split}...")
    
    rejected_count = 0
    
    for f in tqdm(files):
        lane_path = os.path.join(lanes_dir, f)
        vis_path = os.path.join(vis_dir, f)
        
        mask = cv2.imread(lane_path, cv2.IMREAD_GRAYSCALE)
        if mask is None or np.sum(mask) == 0: continue
            
        total_pixels = np.sum(mask > 0)
        is_too_dense = total_pixels > 6000
        
        num_labels, _ = cv2.connectedComponents(mask, connectivity=8)
        object_count = num_labels - 1
        is_too_fragmented = object_count > 8
        
        if is_too_dense or is_too_fragmented:
            blank_mask = np.zeros_like(mask)
            cv2.imwrite(lane_path, blank_mask)
            
            if os.path.exists(vis_path):
                os.remove(vis_path)
                
            rejected_count += 1
            
    print(f"--> Automatically rejected and blanked {rejected_count} corrupted IDD masks.")

def sync_manual_deletions(data_split="data/processed/train"):
    lanes_dir = os.path.join(data_split, "masks_lanes")
    vis_dir = os.path.join(data_split, "visualizations_lanes")
    
    if not os.path.exists(lanes_dir) or not os.path.exists(vis_dir): return
        
    # ONLY evaluate files that belong to the IDD dataset
    masks = set([f for f in os.listdir(lanes_dir) if "idd_lite" in f])
    vis = set([f for f in os.listdir(vis_dir) if "idd_lite" in f])
    
    deleted_by_user = masks - vis
    
    if len(deleted_by_user) == 0: return
        
    for f in deleted_by_user:
        lane_path = os.path.join(lanes_dir, f)
        
        mask = cv2.imread(lane_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None and np.sum(mask) > 0:
            blank_mask = np.zeros_like(mask)
            cv2.imwrite(lane_path, blank_mask)
            
    print(f"--> Synced {len(deleted_by_user)} manual human deletions for IDD in {data_split}.")

if __name__ == "__main__":
    clean_dataset("data/processed/train")
    clean_dataset("data/processed/val")
    sync_manual_deletions("data/processed/train")
    sync_manual_deletions("data/processed/val")