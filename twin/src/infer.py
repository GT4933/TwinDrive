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
    def __init__(self, weights_path="twin/weights/best_model.pth", device=None, lane_thresh=0.50):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lane_thresh = lane_thresh
        print(f"\nInitializing ADAS Pipeline on Hardware: {self.device}")

        # Load TwinLiteNet Model
        self.model = TwinLiteNet().to(self.device)
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"  --> Successfully loaded model weights from: '{weights_path}'")
        else:
            print(f"  [Warning] Checkpoint '{weights_path}' not found! Running with initialized weights.")

        self.model.eval()
        self.trajectory_engine = TrajectoryEngine(image_shape=(640, 360))
        self.hud_visualizer = HUDVisualizer()

    def _extract_prob(self, logits):
        """Robust activation extraction handling single-channel or multi-channel outputs."""
        if logits.shape[1] == 1:
            return torch.sigmoid(logits).squeeze().cpu().numpy()
        return torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

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

        # Forward pass
        start_time = time.time()
        with torch.no_grad():
            drivable_logits, lane_logits = self.model(tensor_img)
        inference_time = time.time() - start_time
        fps = 1.0 / max(inference_time, 1e-5)

        # Compute Class Probabilities
        drivable_prob = self._extract_prob(drivable_logits)
        lane_prob = self._extract_prob(lane_logits)

        # Thresholding (using adaptive lane threshold for early checkpoint convergence)
        drivable_mask = (drivable_prob > 0.35).astype(np.uint8)
        lane_mask = (lane_prob > self.lane_thresh).astype(np.uint8)

        # Trajectory & Geometry Post-Processing
        trajectory_data = self.trajectory_engine.process_frame(drivable_mask, lane_mask)

        # Draw HUD Overlay
        hud_frame = self.hud_visualizer.draw_hud(resized_frame, drivable_mask, lane_mask, trajectory_data, fps=fps)

        # Upscale back to source resolution if necessary
        if (orig_w, orig_h) != (640, 360):
            hud_frame = cv2.resize(hud_frame, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return hud_frame, trajectory_data, fps


def run_batch_inference(
    source_dir="data/processed/val/images",
    output_dir="outputs/predictions",
    max_samples=10,
    random_sample=False,
    bdd_only=False,
    lane_thresh=0.50,
):
    """Runs batch inference with sampling options and configurable lane sensitivity."""
    os.makedirs(output_dir, exist_ok=True)
    pipeline = ADASInferencePipeline(weights_path="twin/weights/best_model.pth", lane_thresh=lane_thresh)

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

    for i, fname in enumerate(selected_files):
        img_path = os.path.join(source_dir, fname)
        raw_img = cv2.imread(img_path)

        if raw_img is None:
            continue

        annotated_frame, traj_data, fps = pipeline.process_image(raw_img)

        out_path = os.path.join(output_dir, f"hud_{fname}")
        cv2.imwrite(out_path, annotated_frame)

        print(
            f"[{i+1:02d}/{len(selected_files):02d}] Processed '{fname}' | "
            f"Speed: {fps:.1f} FPS | "
            f"Mode: {traj_data['mode']} | "
            f"Offset: {traj_data['offset_cm']:.1f} cm ({traj_data['direction']}) -> Saved to '{out_path}'"
        )

    print("=" * 65)
    print(f"Inference complete! Annotated HUD frames saved inside: '{output_dir}/'")
    print("=" * 65)

def run_video_inference(
    source="0",
    output_path=None,
    show=False,
    max_frames=None,
    lane_thresh=0.50,
    weights_path="twin/weights/best_model.pth",
):
    """Runs the ADAS pipeline frame-by-frame on a video file or webcam.

    source      : path to a video file, or a camera index such as "0" for a webcam.
    output_path : where to save the annotated .mp4 (None -> auto-named in twin/outputs/videos/).
    show        : open a live preview window (press 'q' or ESC to stop).
    max_frames  : stop after N frames (handy for quick tests).
    """
    is_webcam = str(source).isdigit()
    cap = cv2.VideoCapture(int(source) if is_webcam else source)
    if not cap.isOpened():
        print(f"Could not open video source '{source}'!")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps != src_fps or src_fps <= 0:  # 0 / NaN on some webcams
        src_fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0

    if output_path is None:
        name = "webcam" if is_webcam else os.path.splitext(os.path.basename(source))[0]
        output_path = os.path.join("twin/outputs/videos", f"{name}_hud.mp4")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (width, height))
    if not writer.isOpened():
        print(f"Could not open VideoWriter for '{output_path}' (codec problem?).")
        cap.release()
        return

    pipeline = ADASInferencePipeline(weights_path=weights_path, lane_thresh=lane_thresh)

    print("=" * 65)
    print(f"        RUNNING ADAS VIDEO INFERENCE: {source}")
    print(f"        {width}x{height} @ {src_fps:.1f} FPS" + (f" | {total} frames" if total else ""))
    print("=" * 65)

    frame_idx = 0
    infer_fps_sum = 0.0
    t_start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of file / camera dropped

            annotated, traj_data, fps = pipeline.process_image(frame)
            writer.write(annotated)

            frame_idx += 1
            infer_fps_sum += fps

            if frame_idx % 30 == 0 or frame_idx == 1:
                progress = f"{frame_idx}/{total}" if total else f"{frame_idx}"
                print(
                    f"[{progress}] Speed: {fps:.1f} FPS | Mode: {traj_data['mode']} | "
                    f"Offset: {traj_data['offset_cm']:.1f} cm ({traj_data['direction']})"
                )

            if show:
                cv2.imshow("TwinDrive ADAS HUD (q / ESC to quit)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("Stopped by user.")
                    break

            if max_frames and frame_idx >= max_frames:
                break
    except KeyboardInterrupt:
        print("Interrupted by user - saving what has been processed so far.")
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    elapsed = max(time.time() - t_start, 1e-5)
    print("=" * 65)
    if frame_idx:
        print(f"Processed {frame_idx} frames in {elapsed:.1f}s "
              f"(end-to-end {frame_idx / elapsed:.1f} FPS, "
              f"avg model-only {infer_fps_sum / frame_idx:.1f} FPS)")
    print(f"Annotated video saved to: '{output_path}'")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TwinLiteNet Real-Time ADAS Telemetry Engine")
    parser.add_argument("--source", type=str, default="data/processed/val/images")
    parser.add_argument("--output", type=str, default="twin/outputs/predictions")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--random", action="store_true", help="Sample images randomly rather than alphabetically")
    parser.add_argument("--bdd-only", action="store_true", help="Filter for BDD100K images (exclude IDD Lite)")
    parser.add_argument("--lane-thresh", type=float, default=0.50, help="Confidence threshold for lane extraction")
    parser.add_argument("--video", type=str, default=None, help="Path to an input video file (runs video mode)")
    parser.add_argument("--webcam", action="store_true", help="Use webcam (camera index 0) as the video source")
    parser.add_argument("--video-out", type=str, default=None, help="Output path for the annotated .mp4")
    parser.add_argument("--show", action="store_true", help="Show a live preview window while processing video")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (quick tests)")

    args = parser.parse_args()

    if args.video or args.webcam:
        run_video_inference(
            source="0" if args.webcam else args.video,
            output_path=args.video_out,
            show=args.show,
            max_frames=args.max_frames,
            lane_thresh=args.lane_thresh,
        )
        sys.exit(0)

    run_batch_inference(
        source_dir=args.source,
        output_dir=args.output,
        max_samples=args.samples,
        random_sample=args.random,
        bdd_only=args.bdd_only,
        lane_thresh=args.lane_thresh,
    )