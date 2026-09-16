# Base Paper Reproduction — TwinLiteNet

This directory holds our reproduction of the base paper:

> Che, Q.-H., Nguyen, D.-P., Pham, M.-Q., Lam, D.-K. "TwinLiteNet: An Efficient and Lightweight Model for Driveable Area and Lane Segmentation in Self-Driving Cars." 2023 International Conference on Multimedia Analysis and Pattern Recognition (MAPR). https://github.com/chequanghuy/TwinLiteNet

Kept separate from `twin/src/` (our own custom `TwinDrive` implementation) so it's clear which architecture/training recipe is the original paper's and which is ours. Scope is BDD100K only, matching the base paper.

## Status
Code is a faithful port of upstream (`const.py`, `model/TwinLite.py`, `dataset.py`, `iou_eval.py`, `loss.py`, `utils.py`, `train.py`, `val.py`, `test_image.py`). The only deviations from upstream are file paths — upstream hardcodes an absolute path from the original author's machine; everything else (architecture, loss weights, hyperparameters, CLI flags) is unchanged. Not yet trained.

## Running
All scripts assume `base_paper/` is the working directory (same convention as upstream's own `python3 train.py`):
```
cd base_paper
../.venv/Scripts/python.exe train.py --max_epochs 2 --batch_size 8   # smoke test
../.venv/Scripts/python.exe train.py                                  # full run (100 epochs / batch 16, paper defaults)
../.venv/Scripts/python.exe val.py --weight test_/checkpoint.pth.tar
../.venv/Scripts/python.exe test_image.py   # drop test images into base_paper/images/ first
```

## Data layout
```
data/bdd100k/
  images/{train,val}     -> junctioned to ../../../datasets/100k/{train,val}        (70000 / 10000)
  segments/{train,val}   -> junctioned to ../../../datasets/bdd_seg_gt/{train,val}  (70000 / 10000, drivable-area masks)
  lane/{train,val}       -> junctioned to ../../../datasets/bdd_lane_gt/{train,val} (70000 / 10000, lane-line masks)
```
All three sources share BDD100K's full 100K-image train/val split with matching filenames (`images/train/X.jpg` <-> `segments/train/X.png` <-> `lane/train/X.png`) — verified 1:1, no renaming needed.

`pretrained/` is for checkpoints (gitignored, empty for now). `results/` (test_image.py output) and `test_/` (train.py's default checkpoint dir) are also gitignored.
