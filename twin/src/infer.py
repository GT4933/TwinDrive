import os
import sys
import time
import argparse
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add project root path for module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.twinlitenet import TwinLiteNet
from src.models.twindrive_v2 import TwinDriveV2, TwinDriveV2LaneFix
from src.postprocess.trajectory import TrajectoryEngine

ARCHS = {"v1": TwinLiteNet, "v2": TwinDriveV2, "v2lf": TwinDriveV2LaneFix}


class HUDVisualizer:
    """Renders semi-transparent drivable area, dynamic trajectory lines, and live telemetry text."""
    def __init__(self, alpha=0.45):
        self.alpha = alpha

    def draw_hud(self, frame, drivable_mask, lane_mask, trajectory_data, fps=0.0, ego_mask=None):
        height, width, _ = frame.shape
        overlay = frame.copy()

        # 1. Semi-transparent Emerald Green overlay for drivable area corridor
        green_mask = np.zeros_like(frame)
        green_mask[drivable_mask > 0] = [0, 210, 80]  # BGR Emerald Green
        cv2.addWeighted(green_mask, self.alpha, overlay, 1.0, 0, overlay)

        # 1b. Blue tint over the predicted ego lane (v2 models only)
        if ego_mask is not None:
            ego_overlay = np.zeros_like(frame)
            ego_overlay[ego_mask > 0] = [200, 90, 0]  # BGR Blue
            cv2.addWeighted(ego_overlay, 0.5, overlay, 1.0, 0, overlay)

        # 2. Distinct Red/Magenta overlay for segmented lane markings
        lane_overlay = np.zeros_like(frame)
        lane_overlay[lane_mask > 0] = [0, 0, 255]  # BGR Red
        cv2.addWeighted(lane_overlay, 0.75, overlay, 1.0, 0, overlay)

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
        cv2.rectangle(overlay, (10, 10), (410, 105), (15, 23, 42), -1)
        cv2.rectangle(overlay, (10, 10), (410, 105), (51, 65, 85), 1)

        # Render Live Telemetry Text Metrics
        mode_str = trajectory_data["mode"]
        offset_val = abs(trajectory_data["offset_cm"])
        direction_str = trajectory_data["direction"]

        mode_color = (0, 255, 120) if mode_str == "LANE_GUIDANCE" else (0, 180, 255)

        cv2.putText(overlay, f"MODE: {mode_str}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1, cv2.LINE_AA)
        cv2.putText(overlay, f"OFFSET: {offset_val:.1f} cm ({direction_str})", (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (241, 245, 249), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"INFERENCE SPEED: {fps:.1f} FPS", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (56, 189, 248), 1, cv2.LINE_AA)

        return overlay


class ADASInferencePipeline:
    """Executes end-to-end model inference, trajectory post-processing, and HUD generation."""
    def __init__(self, weights_path="twin/weights/best_model.pth", device=None, lane_thresh=0.50, ema_alpha=0.0,
                 use_ego=False):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lane_thresh = lane_thresh
        # v2's ego-lane head is opt-in: by default the HUD and trajectory use only drivable area + lanes
        self.use_ego = use_ego
        print(f"\nInitializing ADAS Pipeline on Hardware: {self.device}")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Checkpoint '{weights_path}' not found (refusing to run a randomly initialised model)")
        checkpoint = torch.load(weights_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Training checkpoints record their architecture; older ones predate v2 and are v1
            arch = checkpoint.get("args", {}).get("arch", "v1")
            state = checkpoint["model_state_dict"]
        else:
            arch, state = "v1", checkpoint
        self.model = ARCHS[arch]().to(self.device)
        self.model.load_state_dict(state)
        print(f"  --> Successfully loaded {arch} model weights from: '{weights_path}'")

        self.model.eval()

        # Warm-up: the first CUDA calls include kernel selection and memory setup, so time only after these
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 360, 640, device=self.device)
            for _ in range(10):
                self.model(dummy)
        self._sync()
        self.trajectory_engine = TrajectoryEngine(image_shape=(640, 360), ema_alpha=ema_alpha)
        self.hud_visualizer = HUDVisualizer()

    def _sync(self):
        """CUDA runs asynchronously; wait for queued GPU work so timings measure real compute."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _extract_prob(self, logits):
        """Heads output (1, 1, H, W) logits; sigmoid gives per-pixel probability, returned as (H, W)."""
        return torch.sigmoid(logits)[0, 0].float().cpu().numpy()

    def process_image(self, bgr_image):
        orig_h, orig_w, _ = bgr_image.shape

        # Resize to network input resolution (640x360)
        resized_frame = cv2.resize(bgr_image, (640, 360), interpolation=cv2.INTER_LINEAR)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Normalize and convert to tensor
        norm_img = rgb_frame.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (norm_img - mean) / std

        tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # Forward pass (model-only latency, synchronised so the GPU has actually finished)
        self._sync()
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(tensor_img)
        self._sync()
        inference_time = time.perf_counter() - start_time
        fps = 1.0 / max(inference_time, 1e-5)

        # Compute Class Probabilities
        drivable_prob = self._extract_prob(outputs[0])
        lane_prob = self._extract_prob(outputs[1])

        # Thresholding (0.5 = the decision boundary the loss was trained with)
        drivable_mask = (drivable_prob > 0.5).astype(np.uint8)
        lane_mask = (lane_prob > self.lane_thresh).astype(np.uint8)
        # v2 also predicts the ego lane; keep it inside the drivable area
        ego_mask = ((self._extract_prob(outputs[2]) > 0.5) & (drivable_mask > 0)).astype(np.uint8) \
            if self.use_ego and len(outputs) > 2 else None

        # Trajectory & Geometry Post-Processing
        trajectory_data = self.trajectory_engine.process_frame(drivable_mask, lane_mask, ego_mask, lane_prob=lane_prob)

        # Draw HUD Overlay
        hud_frame = self.hud_visualizer.draw_hud(resized_frame, drivable_mask, lane_mask, trajectory_data, fps=fps,
                                                 ego_mask=ego_mask)

        # Upscale back to source resolution if necessary
        if (orig_w, orig_h) != (640, 360):
            hud_frame = cv2.resize(hud_frame, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return hud_frame, trajectory_data, fps


def run_batch_inference(
    source_dir="data/processed/val/images",
    output_dir="twin/outputs/predictions",
    max_samples=10,
    random_sample=False,
    bdd_only=False,
    lane_thresh=0.50,
    weights_path="twin/weights/best_model.pth",
    use_ego=False,
):
    """Runs batch inference with sampling options and configurable lane sensitivity."""
    os.makedirs(output_dir, exist_ok=True)
    pipeline = ADASInferencePipeline(weights_path=weights_path, lane_thresh=lane_thresh, use_ego=use_ego)

    if not os.path.exists(source_dir):
        print(f"Source path '{source_dir}' does not exist!")
        return

    # Check if the user passed a direct file path instead of a folder
    if os.path.isfile(source_dir):
        target_dir = os.path.dirname(source_dir)
        fname = os.path.basename(source_dir)
        selected_files = [fname]
        source_dir = target_dir  # Reassign so the loop builds the path correctly
    else:
        # Standard directory batch processing
        all_files = [f for f in os.listdir(source_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        if bdd_only:
            all_files = [f for f in all_files if not f.startswith("idd_lite")]

        if not all_files:
            print(f"No valid image files matched in '{source_dir}'!")
            return

        if random_sample:
            selected_files = random.sample(all_files, min(max_samples, len(all_files)))
        else:
            selected_files = sorted(all_files)[:max_samples]

    print("=" * 65)
    print(f"        RUNNING ADAS TELEMETRY INFERENCE ON {len(selected_files)} FRAMES")
    print("=" * 65)

    latencies = []
    for i, fname in enumerate(selected_files):
        img_path = os.path.join(source_dir, fname)
        raw_img = cv2.imread(img_path)

        if raw_img is None:
            continue

        annotated_frame, traj_data, fps = pipeline.process_image(raw_img)
        latencies.append(1000.0 / fps)

        out_path = os.path.join(output_dir, f"hud_{fname}")
        cv2.imwrite(out_path, annotated_frame)

        print(
            f"[{i+1:02d}/{len(selected_files):02d}] Processed '{fname}' | "
            f"Speed: {fps:.1f} FPS | "
            f"Mode: {traj_data['mode']} | "
            f"Offset: {traj_data['offset_cm']:.1f} cm ({traj_data['direction']}) -> Saved to '{out_path}'"
        )

    print("=" * 65)
    if latencies:
        mean_ms = sum(latencies) / len(latencies)
        print(f"Model-only latency: {mean_ms:.2f} ms/frame mean (~{1000.0 / mean_ms:.0f} FPS), "
              f"batch 1, fp32, {pipeline.device}")
    print(f"Inference complete! Annotated HUD frames saved inside: '{output_dir}/'")
    print("=" * 65)


def run_video_inference(video_path, output_path, weights_path="twin/weights/best_model.pth", lane_thresh=0.50,
                        ema_alpha=0.4, use_ego=False):
    """Runs the HUD over a video; consecutive frames are temporally smoothed (ema_alpha = weight of the newest frame)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video '{video_path}'")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipeline = ADASInferencePipeline(weights_path=weights_path, lane_thresh=lane_thresh, ema_alpha=ema_alpha,
                                     use_ego=use_ego)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    writer, latencies, modes = None, [], {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, traj_data, fps = pipeline.process_image(frame)
        if writer is None:
            h, w = annotated.shape[:2]
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
        writer.write(annotated)
        latencies.append(1000.0 / fps)
        modes[traj_data["mode"]] = modes.get(traj_data["mode"], 0) + 1
    cap.release()
    if writer is not None:
        writer.release()

    print("=" * 65)
    if latencies:
        mean_ms = sum(latencies) / len(latencies)
        print(f"{len(latencies)} frames | model-only latency {mean_ms:.2f} ms/frame (~{1000.0 / mean_ms:.0f} FPS)")
        print("Modes: " + ", ".join(f"{m} {n}" for m, n in modes.items()))
    print(f"Annotated video saved to '{output_path}'")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TwinLiteNet Real-Time ADAS Telemetry Engine")
    parser.add_argument("--source", type=str, default="data/processed/val/images")
    parser.add_argument("--output", type=str, default="twin/outputs/predictions")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--random", action="store_true", help="Sample images randomly rather than alphabetically")
    parser.add_argument("--bdd-only", action="store_true", help="Filter for BDD100K images (exclude IDD Lite)")
    parser.add_argument("--lane-thresh", type=float, default=0.50, help="Confidence threshold for lane extraction")
    parser.add_argument("--weights", type=str, default="twin/weights/best_model.pth")
    parser.add_argument("--video", type=str, default=None, help="Process a video file instead of images")
    parser.add_argument("--smooth", type=float, default=0.4,
                        help="Video only: EMA weight of the newest frame for the trajectory (0 = no smoothing)")
    parser.add_argument("--use-ego", action="store_true",
                        help="v2 only: draw the ego-lane prediction and use it as the trajectory fallback")

    args = parser.parse_args()

    if args.video:
        out = args.output if args.output.lower().endswith(".mp4") else os.path.join(args.output, "hud_video.mp4")
        run_video_inference(args.video, out, weights_path=args.weights, lane_thresh=args.lane_thresh,
                            ema_alpha=args.smooth, use_ego=args.use_ego)
        sys.exit(0)

    run_batch_inference(
        source_dir=args.source,
        output_dir=args.output,
        max_samples=args.samples,
        random_sample=args.random,
        bdd_only=args.bdd_only,
        lane_thresh=args.lane_thresh,
        weights_path=args.weights,
        use_ego=args.use_ego,
    )