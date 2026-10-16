import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

def load_teacher_model():
    print("Loading pre-trained YOLOP teacher model from PyTorch Hub...")
    model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device

def process_image(img_path, drivable_path, model, device):
    # --- 1. PREPARATION ---
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None, 0.0
    
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    drivable_mask = cv2.imread(drivable_path, cv2.IMREAD_GRAYSCALE)
    if drivable_mask is None:
        return None, 0.0
    
    # Erode drivable mask:
    # - Moderate erosion (9x9) for YOLOP to keep valid edge lanes while dropping outer curbs
    # - Aggressive erosion (25x15) for OpenCV color extraction so dirt shoulders are excluded
    kernel_yolop = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    eroded_for_yolop = cv2.erode(drivable_mask, kernel_yolop, iterations=1)
    
    kernel_opencv = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 15))
    eroded_for_opencv = cv2.erode(drivable_mask, kernel_opencv, iterations=1)
    
    # --- 2. YOLOP STRUCTURAL EXTRACTION ---
    img_resized = cv2.resize(img_rgb, (640, 640))
    tensor = transforms.ToTensor()(img_resized)
    tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
    tensor = tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, _, ll_seg_out = model(tensor)
        
    yolop_mask = torch.argmax(ll_seg_out, dim=1).squeeze().cpu().numpy()
    yolop_mask = (yolop_mask * 255).astype(np.uint8)
    yolop_mask = cv2.resize(yolop_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    yolop_mask = cv2.bitwise_and(yolop_mask, yolop_mask, mask=eroded_for_yolop)
    
    # --- 3. OPENCV LOCAL-CONTRAST & PIXEL EXTRACTION ---
    # A. Top-Hat Transform: Only extracts thin bright ridges (width <= 19px) that stand out 
    # against darker surrounding asphalt. Wide uniform dirt shoulders cancel out to 0!
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, tophat_kernel)
    _, local_contrast_mask = cv2.threshold(tophat, 25, 255, cv2.THRESH_BINARY)
    
    # B. Tightened HLS Color Mask (Rejects brown/ochre soil by requiring high Lightness)
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(hls, np.array([0, 195, 0]), np.array([180, 255, 70]))
    yellow_mask = cv2.inRange(hls, np.array([15, 135, 110]), np.array([32, 255, 255]))
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # Require OpenCV pixels to be BOTH valid paint color AND locally brighter than their surroundings
    opencv_mask = cv2.bitwise_and(color_mask, local_contrast_mask)
    
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opencv_mask = cv2.morphologyEx(opencv_mask, cv2.MORPH_OPEN, kernel_clean)
    
    # Restrict OpenCV hits strictly to the inner asphalt core
    opencv_mask = cv2.bitwise_and(opencv_mask, opencv_mask, mask=eroded_for_opencv)
    
    # --- 4. THE HYBRID MERGE ---
    raw_hybrid_mask = cv2.bitwise_or(yolop_mask, opencv_mask)
    
    # --- 5. MINIMUM AREA RECTANGLE & SOLIDITY FILTERING ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_hybrid_mask, connectivity=8)
    cleaned_lane_mask = np.zeros_like(raw_hybrid_mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Drop small noise specks
        if area < 35:
            continue
            
        y_coords, x_coords = np.where(labels == i)
        pts = np.column_stack((x_coords, y_coords)).astype(np.int32)
        
        rect = cv2.minAreaRect(pts)
        box_w, box_h = rect[1]
        cy = rect[0][1]
        
        thickness = min(box_w, box_h)
        length = max(box_w, box_h)
        box_area = max(box_w * box_h, 1.0)
        
        # 1. Solidity Check: Real lane segments are ribbon-like and fill their rotated bounding box.
        # Jagged, sprawling dirt patches have very low solidity (< 0.30).
        solidity = area / box_area
        if solidity < 0.32:
            continue
            
        # 2. Tightened Perspective Thickness (640x360 resolution):
        # True lanes are ~6px near the horizon and max ~22px near the bottom bumper.
        y_ratio = cy / float(orig_h)
        max_thickness = max(6.0, 22.0 * y_ratio)
        
        # 3. Elongation & Length Constraints
        aspect_ratio = length / max(thickness, 1.0)
        if thickness > max_thickness or length < 28 or aspect_ratio < 3.0:
            continue
            
        # 4. Reject horizontal stop-lines / speed bumps
        dx = stats[i, cv2.CC_STAT_WIDTH]
        dy = stats[i, cv2.CC_STAT_HEIGHT]
        if dx > dy * 3.0:
            continue
            
        cleaned_lane_mask[labels == i] = 255
            
    # Require at least 120 valid lane pixels to consider the image annotated
    has_valid_lanes = 1.0 if np.sum(cleaned_lane_mask > 0) > 120 else 0.0
    
    if not has_valid_lanes:
        cleaned_lane_mask = np.zeros_like(raw_hybrid_mask)
        
    return cleaned_lane_mask, has_valid_lanes

def run_extraction(data_split, model, device):
    img_dir = os.path.join(data_split, "images")
    drivable_dir = os.path.join(data_split, "masks_drivable")
    lanes_dir = os.path.join(data_split, "masks_lanes")
    
    if not os.path.exists(img_dir):
        return
    
    files = [f for f in os.listdir(img_dir) if "idd_lite" in f]
    print(f"\nProcessing {len(files)} IDD samples in {data_split}...")
    
    annotated_count = 0
    for f in tqdm(files):
        img_path = os.path.join(img_dir, f)
        drivable_path = os.path.join(drivable_dir, f)
        lane_path = os.path.join(lanes_dir, f)
        
        lane_mask, has_lanes = process_image(img_path, drivable_path, model, device)
        
        if lane_mask is not None:
            cv2.imwrite(lane_path, lane_mask)
            if has_lanes:
                annotated_count += 1
            
    print(f"--> Extracted hybrid lane markings for {annotated_count}/{len(files)} IDD samples.")

def save_visualizations(data_split="data/processed/train"):
    img_dir = os.path.join(data_split, "images")
    lane_dir = os.path.join(data_split, "masks_lanes")
    vis_dir = os.path.join(data_split, "visualizations_lanes")
    
    os.makedirs(vis_dir, exist_ok=True)
    
    for old_f in os.listdir(vis_dir):
        if "idd_lite" in old_f:
            os.remove(os.path.join(vis_dir, old_f))
            
    annotated_files = [f for f in os.listdir(lane_dir) if "idd_lite" in f]
    
    print(f"\nSaving visualization overlays for active IDD masks to {vis_dir}...")
    
    saved_vis = 0
    for f in tqdm(annotated_files):
        lane = cv2.imread(os.path.join(lane_dir, f), cv2.IMREAD_GRAYSCALE)
        
        if lane is None or np.sum(lane) == 0: 
            continue
            
        img = cv2.imread(os.path.join(img_dir, f))
        if img is None:
            continue
            
        overlay = img.copy()
        overlay[lane > 0] = [0, 255, 0] 
        
        cv2.imwrite(os.path.join(vis_dir, f), overlay)
        saved_vis += 1
        
    print(f"--> Saved {saved_vis} active visualizations in {vis_dir}.")

if __name__ == "__main__":
    print("==================================================")
    print("      HYBRID PSEUDO-LABEL EXTRACTION TOOL         ")
    print("==================================================")
    
    teacher_model, compute_device = load_teacher_model()
    
    run_extraction("data/processed/train", teacher_model, compute_device)
    run_extraction("data/processed/val", teacher_model, compute_device)
    
    save_visualizations("data/processed/train")
    save_visualizations("data/processed/val")