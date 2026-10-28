# TwinDrive

Lightweight drivable-area and lane segmentation for structured (BDD100K) and unstructured (IDD, Indian) roads, with a lane-offset HUD on top.

TwinDrive builds on [TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet). The base paper is re-implemented in `base_paper/` as our baseline.

From one dashcam frame, the model predicts:
- **drivable area:** road the car can drive on;
- **lane lines:** painted markings and kerbs, as BDD100K labels them;
- **ego lane** (v2 models): the lane the car is currently in.

A post-processing step fits the lane boundaries and estimates the car's lateral offset for a HUD.

---

## Results

Scored by `twin/src/evaluate.py` on 10,204 validation images (10,000 BDD100K + 204 IDD-Lite). Confusion matrices are summed over the whole set, and lanes are scored on BDD only. Single runs.

| Model | Epochs | BDD drivable mIoU | BDD lane IoU | IDD drivable mIoU | Ego-lane mIoU | Params | Conv GMACs |
|---|---|---|---|---|---|---|---|
| **TwinDrive v2lf + YOLOP IDD labels** (shipped checkpoint) | 10 | **0.897** | **0.219** | **0.871** | **0.880** | 360K | 1.78 |
| TwinDrive v2lf + YOLOP IDD labels | 3 | 0.885 | 0.201 | 0.860 | 0.863 | 360K | 1.78 |
| TwinDrive v2 | 3 | 0.887 | 0.171 | 0.851 | 0.865 | 358K | 1.66 |
| TwinDrive v1 | 3 | 0.881 | 0.159 | 0.863 | — | 329K | 1.74 |
| TwinLiteNet baseline (`base_paper/`) | 3 | 0.891 | 0.214 | 0.826* | — | 440K | 3.72 |

\* The baseline never trained on IDD.

**Caveats:**
- The baseline has only been trained for 3 epochs so far, so the 10-epoch row is not yet a like-for-like comparison.
- The cm lane offset on the HUD is **uncalibrated** (an assumed camera geometry); treat it as a demo.
- IDD has no lane labels, so there is no IDD lane score.

The full write-up is in `docs/TwinDrive_Results_Report.md`, if present.

---

## Repository layout

```
TwinDrive/
├── README.md
├── setup.py                     environment check + folder skeleton (not a packaging file)
├── base_paper/                  TwinLiteNet re-implementation (baseline), runs from inside this folder
├── twin/
│   ├── requirements.txt
│   ├── weights_v2lf_yolop_10ep/ shipped best checkpoint (best_model.pth) + train_log.csv
│   └── src/
│       ├── models/
│       │   ├── twindrive_v1.py      TwinDrive v1 (stride 16, single x16 upsample)        --arch v1 / twin
│       │   └── twindrive_v2.py      TwinDrive v2 (stride 8, skip decoders, HFF, coordinate
│       │                            attention, ego head) and v2lf (full-res lane head) --arch v2 / v2lf
│       ├── utils/
│       │   ├── prepare_data.py      raw BDD100K + IDD-Lite -> data/processed (640x360, 0/255 masks)
│       │   ├── make_ego_masks.py    BDD "direct" drivable area -> masks_ego (for v2/v2lf)
│       │   ├── make_yolop_lanes.py  IDD lane pseudo-labels from YOLOP + OpenCV paint detection
│       │   ├── make_pseudo_lanes.py IDD lane pseudo-labels from one of our own checkpoints (self-training)
│       │   ├── dataset.py           loading + augmentation
│       │   ├── metrics.py           dataset-level confusion-matrix metrics
│       │   ├── check_processed.py / check_idd_labels.py / reprocess_idd.py   sanity checks and helpers
│       ├── train.py                 training (CLI)
│       ├── evaluate.py              shared evaluator for v1 / v2 / v2lf / base
│       ├── benchmark.py             params, MACs, synchronised GPU latency
│       ├── infer.py                 images / video -> HUD
│       └── postprocess/trajectory.py  lane-boundary fitting + lateral offset
├── datasets/                    (not in git) BDD100K images and labels
└── data/                        (not in git) raw IDD-Lite + processed training data
```

---

## 1. Setup

You need Python 3.10+ (we used 3.12) and an NVIDIA GPU with CUDA for training. CPU works for inference, slowly.

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```
(On Linux/macOS: `source .venv/bin/activate`.)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
```bash
pip install -r twin/requirements.txt
```
```bash
python setup.py
```
`setup.py` prints the Python, PyTorch and CUDA versions and creates the folder skeleton.

---

## 2. Data

Datasets aren't in git. Put them here:

| Path | Content | Source |
|---|---|---|
| `datasets/100k/{train,val}/*.jpg` | BDD100K images (70,000 / 10,000) | BDD100K |
| `datasets/bdd_seg_gt/{train,val}/*.png` | BDD drivable masks | TwinLiteNet's preprocessed labels |
| `datasets/bdd_lane_gt/{train,val}/*.png` | BDD lane masks | TwinLiteNet's preprocessed labels |
| `data/raw/bdd100k_drivable_maps/labels/{train,val}/*_drivable_id.png` | BDD official drivable maps (1 = own lane, 2 = other lanes); used for the ego lane | BDD100K |
| `data/raw/idd20k_lite/{leftImg8bit,gtFine}/{train,val}/...` | IDD-Lite images and labels | IDD |

Then build the processed dataset. Run all commands from the repo root.

```bash
python twin/src/utils/prepare_data.py
```
This writes about 71k train and 10k val image/mask triples to `data/processed/` (about 21 GB). It fails loudly if any label folder is missing.

```bash
python twin/src/utils/make_ego_masks.py
```
Needed only for v2 / v2lf.

```bash
python twin/src/utils/check_processed.py
```
A sanity check. Expect about 70k / 10k BDD and 1,380 / 204 IDD, with most BDD lane masks non-empty.

### Optional: IDD lane pseudo-labels (YOLOP teacher)
IDD has no lane labels. The shipped model was trained with lane pseudo-labels from YOLOP plus OpenCV paint detection, ported from the `g` branch.

1. Get YOLOP's `hubconf.py`, `lib/` and `weights/End-to-end.pth` from <https://github.com/hustvl/YOLOP> into a local folder. A sparse clone of `lib/`, plus downloading the 96 MB `.pth` file directly, avoids the large repo zip.
2. Run:
```bash
python twin/src/utils/make_yolop_lanes.py --yolop-dir path/to/yolop_src
```
This writes `data/processed/train/masks_lanes_pseudo_yolop/`. In our run, 549 of 1,380 IDD images got labels; the rest are ignored in the loss.

---

## 3. Use the shipped model (no training needed)

### Images
```bash
python twin/src/infer.py --weights twin/weights_v2lf_yolop_10ep/best_model.pth --source path/to/images --samples 20 --output twin/outputs/demo
```
- `--source` can be a folder or a single image.
- Add `--random --bdd-only` to sample from the validation set.
- Add `--use-ego` to also draw the own-lane prediction and use it as the trajectory fallback.

### Video
```bash
python twin/src/infer.py --weights twin/weights_v2lf_yolop_10ep/best_model.pth --video path/to/clip.mp4 --output twin/outputs/demo/hud.mp4
```
`--smooth 0.4` (the default) smooths the lane offset across frames; `0` turns smoothing off.

### HUD legend
| Colour | Meaning |
|---|---|
| Green | drivable area |
| Red | lane pixels (painted lines and kerbs) |
| Blue-green tint (`--use-ego`) | own lane |
| Cyan curves | fitted lane boundaries, drawn only when confident |
| Red / green dots + yellow line | car centre, estimated lane centre and the offset between them |

The architecture is read from the checkpoint automatically.

### Score a checkpoint
```bash
python twin/src/evaluate.py --arch v2lf --weights twin/weights_v2lf_yolop_10ep/best_model.pth
```
`--arch` takes `twin` (v1), `v2`, `v2lf` or `base`.

### Size and speed of every architecture
```bash
python twin/src/benchmark.py
```

---

## 4. Train

The configuration of the shipped model:
```bash
python twin/src/train.py --arch v2lf --epochs 10 --workers 12 --geometric-aug --lane-loss tversky --tversky-alpha 0.9 --tversky-beta 0.1 --lane-overlap-weight 1.0 --lane-focal-gamma 2 --pseudo-lanes --pseudo-dir masks_lanes_pseudo_yolop --pseudo-weight 1.0 --out-dir twin/weights_my_run
```
Each run writes `best_model.pth` (best on (BDD drivable mIoU + BDD lane IoU) / 2), `latest_model.pth` (for resuming) and `train_log.csv` to `--out-dir`.

| Option | Meaning |
|---|---|
| `--arch v1 \| v2 \| v2lf` | model (v2lf = v2 with a full-resolution lane head; recommended) |
| `--epochs N` | length of the cosine LR schedule |
| `--batch-size 8`, `--lr 5e-4` | defaults |
| `--workers N` | data-loading processes (12 was fastest on an 8-core CPU) |
| `--geometric-aug` | random scale / shift / rotate |
| `--lane-loss dice \| tversky`, `--tversky-alpha`, `--tversky-beta` | overlap term of the lane loss (α weights false positives) |
| `--lane-overlap-weight`, `--lane-focal-gamma` | lane Dice/Tversky weight; focal BCE strength (0 = plain BCE) |
| `--task-weighting fixed \| uncertainty` | how the task losses are combined |
| `--pseudo-lanes --pseudo-dir <folder> --pseudo-weight w` | supervise IDD lanes with pseudo-labels |
| `--init-weights ckpt` | fine-tune from a checkpoint |
| `--resume latest_model.pth` | continue an interrupted run (same `--epochs` and `--arch`) |
| `--max-train-samples N --max-val-samples N` | quick smoke tests |

**Timing:** about 17 minutes per epoch for v2lf on an RTX 5060 Laptop GPU with 12 workers.

**Before a long run:** keep the laptop plugged in with sleep disabled; earlier runs lost hours to sleep.

---

## 5. Baseline (TwinLiteNet)

Run these from inside `base_paper/`. They expect `base_paper/data/bdd100k/{images,segments,lane}/{train,val}` (junctions or links to the `datasets/` folders). See `base_paper/README.md`.

```bash
cd base_paper
```
```bash
python train.py --max_epochs 3 --num_workers 4 --batch_size 16 --savedir ./test_
```
Then, from the repo root, score it with the same evaluator as TwinDrive:
```bash
python twin/src/evaluate.py --arch base --weights base_paper/test_/model_2.pth
```

---

## Branches

| Branch | Content |
|---|---|
| `main` | everything above |
| `fix/phase0-data` | the working branch that was merged into `main` |
| `g` | GT4933's IDD lane-labelling experiments; the labeller is ported here as `make_yolop_lanes.py` |
| `vid` | early video / webcam inference work |
