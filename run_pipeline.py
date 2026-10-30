"""
TwinDrive Automated End-to-End ADAS Pipeline Orchestrator.

Automates the complete workflow from environment diagnostics to data preparation,
model training, hardware benchmarking, metric evaluation, and HUD telemetry inference.
Supports TwinDrive (v1, v2, v2lf) and the original base_paper TwinLiteNet model.

Usage Examples:
    # 1. Quick demo run (< 30s): verify environment, benchmark hardware, generate sample HUD frames
    python run_pipeline.py --mode demo

    # 2. Evaluation run: benchmark hardware, score validation split, generate HUD predictions
    python run_pipeline.py --mode eval

    # 3. Model training: verify data, extract ego masks, train v2lf model, evaluate and test
    python run_pipeline.py --mode train --arch v2lf --epochs 10

    # 4. Base Paper Baseline: evaluate or run inference using the original TwinLiteNet model
    python run_pipeline.py --arch base --weights base_paper/test_/model_2.pth --mode demo
"""

import os
import sys
import time
import argparse
import subprocess

# Prevent thread contention on Windows
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")


def print_banner(text, fill="="):
    width = 70
    print(f"\n{fill * width}")
    print(f"  {text}")
    print(f"{fill * width}")


def run_command(step_name, command, cwd=None):
    """Executes a pipeline subprocess command with timing and status monitoring."""
    print_banner(f"PIPELINE STEP: {step_name.upper()}")
    print(f"Command : {command}")
    print(f"Working : {cwd or os.getcwd()}")
    start_time = time.time()

    process = subprocess.run(command, shell=True, cwd=cwd)

    elapsed = time.time() - start_time
    if process.returncode != 0:
        print_banner(f"[FAILED] Step '{step_name}' exited with error code {process.returncode} ({elapsed:.1f}s)", fill="!")
        sys.exit(process.returncode)
    else:
        print(f"\n[OK] Step '{step_name}' completed in {elapsed:.1f}s.")
    return elapsed


def check_processed_data():
    """Returns True if normalized data exists in data/processed."""
    img_dir = os.path.join("data", "processed", "train", "images")
    return os.path.isdir(img_dir) and len(os.listdir(img_dir)) > 100


def check_ego_masks():
    """Returns True if ego-lane masks exist in data/processed/train/masks_ego."""
    ego_dir = os.path.join("data", "processed", "train", "masks_ego")
    return os.path.isdir(ego_dir) and len(os.listdir(ego_dir)) > 100


def check_pseudo_lanes(pseudo_dir_name="masks_lanes_pseudo_yolop"):
    """Returns True if pseudo-lane masks exist."""
    pseudo_dir = os.path.join("data", "processed", "train", pseudo_dir_name)
    return os.path.isdir(pseudo_dir) and len(os.listdir(pseudo_dir)) > 50


def main():
    parser = argparse.ArgumentParser(
        description="TwinDrive Automated End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "eval", "train", "full"],
        default="demo",
        help="demo: quick verification & sample HUD; eval: benchmark & score val set; "
             "train: train model, evaluate & infer; full: data prep + pseudo labels + train + eval + infer"
    )
    parser.add_argument(
        "--arch",
        choices=["v1", "v2", "v2lf", "base"],
        default="v2lf",
        help="Model architecture: v1, v2, v2lf, or base (original paper) (default: v2lf)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4, safe for Windows)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights checkpoint (default: shipped v2lf checkpoint, or base_paper model for base)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=15,
        help="Number of sample predictions to generate in inference (default: 15)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional path to a video file (.mp4) for video stream inference"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="twin/outputs/predictions",
        help="Destination directory for HUD prediction frames"
    )
    parser.add_argument(
        "--force-prep",
        action="store_true",
        help="Force re-running data ingestion and normalization"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip metric evaluation step"
    )
    parser.add_argument(
        "--use-ego",
        action="store_true",
        help="Render ego-lane prediction overlay on HUD (v2/v2lf only)"
    )

    args = parser.parse_args()
    py = f'"{sys.executable}"'
    pipeline_start = time.time()

    # Assign smart default weights if not supplied
    if args.weights is None:
        if args.arch == "base":
            for candidate in ("base_paper/test_/model_2.pth", "base_paper/test_/checkpoint.pth.tar"):
                if os.path.exists(candidate):
                    args.weights = candidate
                    break
            if args.weights is None:
                args.weights = "base_paper/test_/model_2.pth"
        else:
            args.weights = "twin/weights_v2lf_yolop_10ep/best_model.pth"

    print_banner(f"STARTING TWINDRIVE PIPELINE [MODE: {args.mode.upper()}]")
    print(f"Target Architecture : {args.arch}")
    print(f"Using Weights       : {args.weights}")
    print(f"Python Environment  : {sys.executable}")

    # -------------------------------------------------------------
    # Step 0: Environment Diagnostics & Directory Scaffolding
    # -------------------------------------------------------------
    run_command("0. Environment Diagnostics", f"{py} setup.py")

    # -------------------------------------------------------------
    # Step 1: Data Ingestion & Normalization (TwinDrive models)
    # -------------------------------------------------------------
    if args.arch != "base" and (args.mode in ("train", "full") or args.force_prep):
        if not check_processed_data() or args.force_prep:
            run_command("1. Data Ingestion & Normalization", f"{py} twin/src/utils/prepare_data.py")
        else:
            print("\n[INFO] Valid data found in data/processed/. Skipping Step 1 (use --force-prep to re-run).")

        # -------------------------------------------------------------
        # Step 2: Ego-Lane Mask Extraction (for v2 / v2lf)
        # -------------------------------------------------------------
        if args.arch in ("v2", "v2lf"):
            if not check_ego_masks() or args.force_prep:
                run_command("2. Ego-Lane Mask Extraction", f"{py} twin/src/utils/make_ego_masks.py")
            else:
                print("\n[INFO] Ego masks found in data/processed/*/masks_ego/. Skipping Step 2.")

    # -------------------------------------------------------------
    # Step 3: Weak Pseudo-Label Generation (Full mode only)
    # -------------------------------------------------------------
    if args.arch != "base" and args.mode == "full":
        if not check_pseudo_lanes():
            run_command("3. YOLOP + Paint Pseudo-Labeling", f"{py} twin/src/utils/make_yolop_lanes.py")
        else:
            print("\n[INFO] Pseudo labels found in masks_lanes_pseudo_yolop/. Skipping Step 3.")

    # -------------------------------------------------------------
    # Step 4: Model Training
    # -------------------------------------------------------------
    if args.mode in ("train", "full"):
        if args.arch == "base":
            train_cmd = f"{py} train.py --max_epochs {args.epochs} --batch_size {args.batch_size} --savedir test_"
            run_command("4. Base Paper Model Training", train_cmd, cwd="base_paper")
            args.weights = os.path.join("base_paper", "test_", f"model_{args.epochs - 1}.pth")
        else:
            out_weights_dir = f"twin/weights_{args.arch}_run"
            train_cmd = [
                f"{py} twin/src/train.py",
                f"--arch {args.arch}",
                f"--epochs {args.epochs}",
                f"--batch-size {args.batch_size}",
                f"--workers {args.workers}",
                "--geometric-aug",
                "--lane-loss tversky",
                "--tversky-alpha 0.9",
                "--tversky-beta 0.1",
                "--lane-overlap-weight 1.0",
                "--lane-focal-gamma 2",
                f"--out-dir {out_weights_dir}"
            ]
            if check_pseudo_lanes():
                train_cmd.append("--pseudo-lanes --pseudo-dir masks_lanes_pseudo_yolop --pseudo-weight 1.0")

            run_command("4. Multi-Task Model Training", " ".join(train_cmd))
            args.weights = os.path.join(out_weights_dir, "best_model.pth")

    # -------------------------------------------------------------
    # Step 5: Hardware Profiling & Evaluation
    # -------------------------------------------------------------
    if args.mode in ("demo", "eval", "train", "full"):
        run_command("5a. Hardware Benchmark & TFLOPS", f"{py} twin/src/benchmark.py")

        if args.mode in ("eval", "train", "full") and not args.skip_eval:
            eval_cmd = f"{py} twin/src/evaluate.py --arch {args.arch} --weights {args.weights} --workers {args.workers}"
            run_command("5b. Full Metric Evaluation", eval_cmd)

    # -------------------------------------------------------------
    # Step 6: Real-Time Inference & HUD Rendering
    # -------------------------------------------------------------
    if args.video:
        video_out = os.path.join(args.output_dir, "annotated_hud_video.mp4")
        infer_cmd = [
            f"{py} twin/src/infer.py",
            f"--arch {args.arch}",
            f"--weights {args.weights}",
            f"--video {args.video}",
            f"--output {video_out}",
            "--smooth 0.4"
        ]
        if args.use_ego:
            infer_cmd.append("--use-ego")
        run_command("6. Video ADAS HUD Inference", " ".join(infer_cmd))
    else:
        infer_cmd = [
            f"{py} twin/src/infer.py",
            f"--arch {args.arch}",
            f"--weights {args.weights}",
            f"--samples {args.samples}",
            "--random",
            f"--output {args.output_dir}"
        ]
        if args.use_ego:
            infer_cmd.append("--use-ego")
        run_command("6. Batch Image ADAS HUD Inference", " ".join(infer_cmd))

    # -------------------------------------------------------------
    # Complete
    # -------------------------------------------------------------
    total_time = time.time() - pipeline_start
    print_banner(f"PIPELINE COMPLETE [SUCCESS] (Total Duration: {total_time:.1f}s)")
    print(f"Processed outputs saved to: '{args.output_dir}'")


if __name__ == "__main__":
    main()
