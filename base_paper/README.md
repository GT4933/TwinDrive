# Base Paper Reproduction — TwinLiteNet

This directory holds our reproduction of the base paper:

> Che, Q.-H., Nguyen, D.-P., Pham, M.-Q., Lam, D.-K. "TwinLiteNet: An Efficient and Lightweight Model for Driveable Area and Lane Segmentation in Self-Driving Cars." 2023 International Conference on Multimedia Analysis and Pattern Recognition (MAPR). https://github.com/chequanghuy/TwinLiteNet

Kept separate from `src/` (our own custom `TwinDrive` implementation) so it's clear which architecture/training recipe is the original paper's and which is ours. Scope is BDD100K only, matching the base paper.

## Status
Data layout is wired up. Code (model, dataset loader, loss, train/val/test scripts) has not been ported yet — pending task assignment within the team.

## Data layout
```
data/bdd100k/
  images/{train,val}     -> junctioned to ../../../100k/{train,val}        (70000 / 10000)
  segments/{train,val}   -> junctioned to ../../../bdd_seg_gt/{train,val}  (70000 / 10000, drivable-area masks)
  lane/{train,val}       -> junctioned to ../../../bdd_lane_gt/{train,val} (70000 / 10000, lane-line masks)
```
All three sources share BDD100K's full 100K-image train/val split with matching filenames (`images/train/X.jpg` <-> `segments/train/X.png` <-> `lane/train/X.png`) — verified 1:1, no renaming needed.

`pretrained/` is for checkpoints (gitignored, empty for now).
