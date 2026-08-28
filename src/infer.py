import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add project root path for seamless module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.twinlitenet import TwinLiteNet
from src.postprocess.trajectory import TrajectoryEngine

class HUDVisualizer:
    """Renders semi-transparent drivable area, dynamic trajectory lines, and live telemetry text."""
    def __init__(self, alpha=0.45):
        self.alpha = alpha

    def draw_hud(self, frame, drivable_mask, lane_mask, trajectory_data, fps=0.0):
        height, width, _ = frame.shape
        overlay = frame.copy()

        # 1. Semi-transparent Emerald Green overlay for drivable area corridor
        green_mask = np.zeros_like(frame)
        green_mask[drivable_mask > 0] = [0, 210, 80]  # BGR Emerald Green
        cv2.addWeighted(green_mask, self.alpha, overlay, 1.0, 0, overlay)

        # 2. Cyan overlay for detected lane lines
        cyan_mask = np.zeros_like(frame)
        cyan_mask[lane_mask > 0] = [255, 200, 0]  # BGR Cyan/Yellow
        cv2.addWeighted(cyan_mask, 0.7, overlay, 1.0, 0, overlay)

        # 3. Draw fitted trajectory polynomial curves
        left_x, left_y = trajectory_data["left_x"], trajectory_data["left_y"]
        right_x, right_y = trajectory_data["right_x"], trajectory_data["right_y"]

        if left_x is not None and left_y is not None:
            pts_left = np.array([np.transpose(np.vstack([left_x, left_y]))], dtype=np.int32)
            cv2.polylines(overlay, pts_left, isClosed=False, color=(255, 255, 0), thickness=4)

        if right_x is not None and right_y is not None:
            pts_right = np.array([np.transpose(np.vstack([right_x, right_y]))], dtype=np.int32)
            cv2.polylines(overlay, pts_right, isClosed=False, color=(255, 255, 0), thickness=4)

        # 4. Draw Center Telemetry Line & Vehicle Camera Reference
        cam_center_x = int(width / 2.0)
        lane_center_x = int(trajectory_data["lane_center_x"])
        bottom_y = height - 20

        # Draw vehicle midpoint anchor (Red)
        cv2.circle(overlay, (cam_center_x, bottom_y), 8, (0, 0, 255), -1)
        # Draw path centerline target anchor (Green)
        cv2.circle(overlay, (lane_center_x, bottom_y), 8, (0, 255, 0), -1)
        # Connecting line
        cv2.line(overlay, (cam_center_x, bottom_y), (lane_center_x, bottom_y), (0, 255, 255), 3)

        # 5. Render Dark Telemetry Dashboard Header Box
        cv2.rectangle(overlay, (10, 10), (380, 100), (15, 23, 42), -1)
        cv2.rectangle(overlay, (10, 10), (380, 100), (51, 65, 85), 1)

        # Render Live Telemetry Text Metrics
        mode_str = trajectory_data["mode"]
        offset_val = abs(trajectory_data["offset_cm"])
        direction_str = trajectory_data["direction"]

        mode_color = (0, 255, 120) if mode_str == "LANE_GUIDANCE" else (0, 180, 255)

        cv2.putText(overlay, f"MODE: {mode_str}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1, cv2.LINE_AA)
        cv2.putText(overlay, f"OFFSET: {offset_val:.1f} cm ({direction_str})", (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (241, 245, 249), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"INFERENCE SPEED: {fps:.1f} FPS", (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (56, 189, 248), 1, cv2.LINE_AA)

        return overlay

class ADASInferencePipeline:
    """Executes end-to-end model inference, trajectory post-processing, and HUD generation."""
    def __init__(self, weights_path="weights/best_model.pth", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nInitializing ADAS Pipeline on Hardware: {self.device}")

        # Load TwinLiteNet Model
        self.model = TwinLiteNet().to(self.device)
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            # Handle dictionary checkpoint vs direct state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"  --> Successfully loaded model weights from: '{weights_path}'")
        else:
            print(f"  [Warning] Checkpoint '{weights_path}' not found! Running with initialized weights.")

        self.model.eval()
        self.trajectory_engine = TrajectoryEngine(image_shape=(640, 360))
        self.hud_visualizer = HUDVisualizer()

    def process_image(self, bgr_image):
        orig_h, orig_w, _ = bgr_image.shape
        
        # Resize to standard network input resolution (640x360)
        resized_frame = cv2.resize(bgr_image, (640, 360), interpolation=cv2.INTER_LINEAR)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Normalize and convert to PyTorch Tensor
        norm_img = rgb_frame.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (norm_img - mean) / std

        tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # Model Forward Pass
        start_time = time.time()
        with torch.no_grad():
            drivable_logits, lane_logits = self.model(tensor_img)
        inference_time = time.time() - start_time
        fps = 1.0 / max(inference_time, 1e-5)

        # Compute Class Probabilities
        drivable_prob = torch.softmax(drivable_logits, dim=1)[0, 1].cpu().numpy()
        lane_prob = torch.softmax(lane_logits, dim=1)[0, 1].cpu().numpy()

        # Binary Mask Thresholding
        drivable_mask = (drivable_prob > 0.5).astype(np.uint8)
        lane_mask = (lane_prob > 0.5).astype(np.uint8)

        # Trajectory & Geometry Post-Processing
        trajectory_data = self.trajectory_engine.process_frame(drivable_mask, lane_mask)

        # Draw HUD Telemetry Overlay
        hud_frame = self.hud_visualizer.draw_hud(resized_frame, drivable_mask, lane_mask, trajectory_data, fps=fps)

        # Upscale back to original input image resolution if different
        if (orig_w, orig_h) != (640, 360):
            hud_frame = cv2.resize(hud_frame, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return hud_frame, trajectory_data, fps

def run_batch_inference(source_dir="data/processed/val/images", output_dir="outputs/predictions", max_samples=10):
    """Runs end-to-end inference over validation images and saves predictions with HUD overlays."""
    os.makedirs(output_dir, exist_ok=True)
    pipeline = ADASInferencePipeline(weights_path="weights/best_model.pth")

    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist! Generating sample output instead.")
        return

    filenames = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:max_samples]

    if not filenames:
        print(f"No validation image files found inside '{source_dir}'!")
        return

    print("=" * 65)
    print(f"        RUNNING ADAS TELEMETRY INFERENCE ON {len(filenames)} FRAMES")
    print("=" * 65)

    for i, fname in enumerate(filenames):
        img_path = os.path.join(source_dir, fname)
        raw_img = cv2.imread(img_path)

        if raw_img is None:
            continue

        annotated_frame, traj_data, fps = pipeline.process_image(raw_img)

        out_path = os.path.join(output_dir, f"hud_{fname}")
        cv2.imwrite(out_path, annotated_frame)

        print(f"[{i+1:02d}/{len(filenames):02d}] Processed '{fname}' | "
              f"Speed: {fps:.1f} FPS | "
              f"Mode: {traj_data['mode']} | "
              f"Offset: {traj_data['offset_cm']:.1f} cm ({traj_data['direction']}) -> Saved to '{out_path}'")

    print("=" * 65)
    print(f"Inference complete! Annotated HUD frames saved inside: '{output_dir}/'")
    print("=" * 65)

if __name__ == "__main__":
    run_batch_inference(max_samples=10)