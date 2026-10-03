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
    if img_bgr is None: return None, 0.0
    
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    drivable_mask = cv2.imread(drivable_path, cv2.IMREAD_GRAYSCALE)
    if drivable_mask is None: return None, 0.0
    
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
    
    # --- 3. OPENCV PIXEL EXTRACTION ---
    road_roi = cv2.bitwise_and(img_bgr, img_bgr, mask=drivable_mask)
    hls = cv2.cvtColor(road_roi, cv2.COLOR_BGR2HLS)
    
    white_mask = cv2.inRange(hls, np.array([0, 200, 0]), np.array([180, 255, 255]))
    yellow_mask = cv2.inRange(hls, np.array([15, 60, 80]), np.array([35, 255, 255]))
    
    opencv_mask = cv2.bitwise_or(white_mask, yellow_mask)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opencv_mask = cv2.morphologyEx(opencv_mask, cv2.MORPH_OPEN, kernel_clean)
    
    # --- 4. THE HYBRID MERGE ---
    raw_hybrid_mask = cv2.bitwise_or(yolop_mask, opencv_mask)
    raw_hybrid_mask = cv2.bitwise_and(raw_hybrid_mask, raw_hybrid_mask, mask=drivable_mask)
    
    # --- 5. MINIMUM AREA RECTANGLE FILTERING (The Perspective Fix) ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_hybrid_mask, connectivity=8)
    cleaned_lane_mask = np.zeros_like(raw_hybrid_mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Drop pixel dust before doing heavy math
        if area < 30:
            continue
            
        # 1. Extract the exact coordinates of the blob
        y_coords, x_coords = np.where(labels == i)
        pts = np.column_stack((x_coords, y_coords))
        
        # 2. Fit an angled bounding box to get the true physical thickness
        # This solves the diagonal line problem completely
        rect = cv2.minAreaRect(pts)
        box_w, box_h = rect[1]
        cy = rect[0][1]
        
        thickness = min(box_w, box_h)
        length = max(box_w, box_h)
        
        # 3. Apply your perspective ratio logic (y_ratio goes from 0.0 top to 1.0 bottom)
        y_ratio = cy / float(orig_h)
        
        # A valid lane is ~10px thick near the horizon, stretching to ~45px at the ego-vehicle bumper
        max_thickness = max(10.0, 45.0 * y_ratio)
        
        # 4. The Structural Kill Switch
        # If the shape is physically thicker than perspective allows, it is a dirt blob/glare.
        if thickness > max_thickness:
            continue
            
        # 5. Length constraint (Must be a continuous spline, not a localized square)
        if length < 30:
            continue
            
        cleaned_lane_mask[labels == i] = 255
            
    # Require at least 150 valid lane pixels to consider the image annotated
    has_valid_lanes = 1.0 if np.sum(cleaned_lane_mask > 0) > 150 else 0.0
    
    return cleaned_lane_mask, has_valid_lanes

def run_extraction(data_split, model, device):
    img_dir = os.path.join(data_split, "images")
    drivable_dir = os.path.join(data_split, "masks_drivable")
    lanes_dir = os.path.join(data_split, "masks_lanes")
    
    if not os.path.exists(img_dir): return
    
    files = [f for f in os.listdir(img_dir) if "idd_lite" in f]
    print(f"\nProcessing {len(files)} IDD samples in {data_split}...")
    
    annotated_count = 0
    for f in tqdm(files):
        img_path = os.path.join(img_dir, f)
        drivable_path = os.path.join(drivable_dir, f)
        lane_path = os.path.join(lanes_dir, f)
        
        lane_mask, has_lanes = process_image(img_path, drivable_path, model, device)
        
        if lane_mask is not None and has_lanes:
            cv2.imwrite(lane_path, lane_mask)
            annotated_count += 1
            
    print(f"--> Extracted hybrid lane markings for {annotated_count}/{len(files)} IDD samples.")

def save_visualizations(data_split="data/processed/train"):
    img_dir = os.path.join(data_split, "images")
    lane_dir = os.path.join(data_split, "masks_lanes")
    vis_dir = os.path.join(data_split, "visualizations_lanes")
    
    os.makedirs(vis_dir, exist_ok=True)
    annotated_files = [f for f in os.listdir(lane_dir) if "idd_lite" in f]
    
    print(f"\nSaving {len(annotated_files)} visualization overlays to {vis_dir}...")
    
    for f in tqdm(annotated_files):
        lane = cv2.imread(os.path.join(lane_dir, f), cv2.IMREAD_GRAYSCALE)
        
        if lane is None or np.sum(lane) == 0: 
            continue
            
        img = cv2.imread(os.path.join(img_dir, f))
        if img is None: continue
            
        overlay = img.copy()
        # Draw the hybrid lane predictions in bright green
        overlay[lane > 0] = [0, 255, 0] 
        
        cv2.imwrite(os.path.join(vis_dir, f), overlay)

if __name__ == "__main__":
    print("==================================================")
    print("      HYBRID PSEUDO-LABEL EXTRACTION TOOL         ")
    print("==================================================")
    
    teacher_model, compute_device = load_teacher_model()
    
    run_extraction("data/processed/train", teacher_model, compute_device)
    run_extraction("data/processed/val", teacher_model, compute_device)
    
    save_visualizations("data/processed/train")
    save_visualizations("data/processed/val")