# MaLDAM: Masked Localized Domain Adaptation for Malaria Detection in Low-Cost Microscopic Images

MaLDAM is a training pipeline that adapts a YOLOv5 malaria-parasite detector from **expensive, high-cost microscope (HCM)** images to **cheap, low-cost microscope (LCM)** images, using a masked, localized contrastive objective alongside the ordinary object-detection loss. It is built on top of [CodaMal](https://arxiv.org/abs/2402.10478) (Dave et al., 2024) and [VICRegL](https://arxiv.org/abs/2210.01571) (Bardes et al., 2022), and extends both by restricting the local contrastive matching to the parasite (foreground) regions of each image pair instead of the whole frame.

> Muditha Fernando, Tishan Rathnasekara, Saeedha Nazar, Avishka Perera, Tharindu Kaluarachchi
> **HemaRAI 2026 (Oral)**

---

## Table of Contents

- [Overview](#overview)
- [Method](#method)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Training](#training)
  - [Single-GPU](#single-gpu)
  - [Multi-GPU (DDP)](#multi-gpu-ddp)
  - [Resuming a Run](#resuming-a-run)
  - [Hyperparameter Evolution](#hyperparameter-evolution)
- [Outputs](#outputs)
- [Cross-Dataset Benchmarking](#cross-dataset-benchmarking)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

Low-cost microscopes (LCM) make malaria screening feasible in resource-constrained clinics, but detectors trained on images from expensive, high-quality microscopes (HCM) generalize poorly to the noisier, lower-resolution optics of an LCM. MaLDAM closes this domain gap **during training**, not as a separate fine-tuning stage: every training step jointly optimizes

1. a standard YOLOv5 detection loss (box, objectness, class) for locating and staging malaria parasites (*ring*, *trophozoite*, *schizont*, *gametocyte*), and
2. a **VICRegL-based contrastive domain-adaptation loss** that pulls the backbone's HCM and LCM feature representations of the *same physical sample* together, with the local (patch-level) term restricted to the parasite-containing regions of the image via a foreground mask.

The result is a single detector that is trained on HCM images (where high-quality labels are easier to obtain) but whose features are explicitly regularized to transfer to LCM images.

## Method

```
                         ┌───────────────────────┐
      training image ───►│   YOLOv5m backbone     │───► detection head ──► box / obj / cls loss
                         │   (shared weights)     │
                         └───────────────────────┘

      HCM crop     ─────►┌───────────────────────┐        ┌───────────────┐
      (paired      │     │   YOLOv5m backbone     │──────►│  VICRegL head  │
       sample)     │     │   (shared weights)     │        │ (global+local) │──► VICRegL loss
                   │     └───────────────────────┘        └───────┬───────┘   (weighted, added
      LCM crop     └────►┌───────────────────────┐                │           to detection loss)
      (same sample)      │   YOLOv5m backbone     │────────────────┘
                         │   (shared weights)     │
                         └───────────────────────┘
                                    ▲
                                    │ local matching restricted to
                          foreground / parasite masks (mh, ml)
```

- **Shared backbone.** The same YOLOv5m backbone extracts features for the detection image and for an HCM/LCM pair of the same underlying sample (paired by filename convention, see [Dataset](#dataset)).
- **VICRegL head** (`src/utils/vicregl/model.py`) projects backbone feature maps into a global embedding (whole-image representation) and a per-location map embedding, per the [VICRegL](https://arxiv.org/abs/2210.01571) formulation.
- **VICRegL loss** (`src/utils/vicregl/loss.py`) combines:
  - a **global** invariance/variance/covariance term over the whole-image embeddings, and
  - a **local** term that nearest-neighbor-matches patch embeddings between the HCM and LCM views and applies the same invariance/variance/covariance criterion to the matches.
  - `vicregl.alpha` (`config/train_contrastive.yaml`) balances the two: `alpha=1.0` is pure global VICReg, `alpha=0.0` is pure local matching.
- **Masking (the "M" in MaLDAM).** Local matches are computed only between patches that fall inside the parasite foreground mask of *both* views (`fg_mask` in `src/contrastive_trainer.py`), so the domain-adaptation signal comes from clinically relevant regions rather than background/stain artifacts.
- **Domain-generalization augmentations** (`src/utils/augmentations.py`): `MalariaDGAugment` (blur, noise, brightness/contrast, color, stain, resolution, JPEG, distortion, vignette) and `FourierDGAugment` (Fourier-domain style jitter) are applied on the detection path to further close the HCM/LCM appearance gap.
- **Combined loss**: `loss_detection + vicregl.loss_weight * loss_VICRegL`, backpropagated jointly each step. Training logs both as `box/obj/cls` and `DAC` (Domain Adaptation Contrastive) losses.

## Repository Structure

```
MaLDAM/
├── main.py                        # entry point: single-GPU and DDP training/evolution
├── config/
│   └── train_contrastive.yaml     # single source of truth for a run (data, model, VICRegL, optimizer, schedule)
├── data/
│   ├── m5_400x.yaml                # M5 dataset split, 400x zoom
│   ├── m5_100x.yaml                # M5 dataset split, 100x zoom
│   └── m5_1000xa.yaml              # M5 dataset split, 1000x zoom
├── src/
│   ├── config.py                   # YAML <-> OmegaConf load/save
│   ├── contrastive_trainer.py      # ContrastiveTrainer: the full training/validation/benchmark loop
│   ├── models/                     # YOLOv5 model definitions (backbone, detection head, yolov5m.yaml)
│   └── utils/
│       ├── dataloaders_contrastive.py  # HCM/LCM-paired dataset + foreground mask loading
│       ├── dataloaders.py              # standard YOLOv5 val/benchmark dataloader
│       ├── vicregl/                    # VICRegL head, loss, nearest-neighbor matching utils
│       ├── feature_matching.py         # nearest-neighbor feature matching primitives
│       ├── augmentations.py            # incl. MalariaDGAugment / FourierDGAugment
│       ├── loss.py                     # YOLOv5 detection loss
│       ├── val.py                      # detection validation / benchmarking loop
│       └── metrics.py, plots.py, ...   # AP/PR computation, plotting
├── requirements.txt
└── yolov5m.pt                       # pretrained YOLOv5m weights used to initialize training
```

## Installation

```bash
git clone <this-repo-url> MaLDAM
cd MaLDAM
python -m venv .venv && source .venv/bin/activate   # or conda create -n maldam python=3.10
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.x with CUDA support, and 1-2 NVIDIA GPUs.

## Dataset

MaLDAM trains on the **M5 malaria microscopy dataset**, which pairs HCM and LCM captures of the same blood-smear field at multiple zoom levels (100x/400x/1000x).

- Download: https://drive.google.com/drive/folders/1k2GuIu6obj3Nz--dOTLuwQnJ2qs1sXxE
- Background / original release notes: https://github.com/intelligentMachines-ITU/LowCostMalariaDetection_CVPR_2022

Expected directory layout (relative to a `path:` root set in the dataset YAML):

```
<path>/
├── Images/
│   ├── HCM/{train,val,test}/{100x,400x,1000x}/*.png
│   └── LCM/{train,val,test}/{100x,400x,1000x}/*.png
├── Labels/            # YOLO-format .txt boxes, mirrored under the same HCM/LCM/split/zoom tree
└── Masks/              # per-image parasite foreground masks (.npy), same tree, used for masked local matching
```

HCM and LCM samples are paired automatically by path convention (`.../HCM/...` ↔ `.../LCM/...`, `HCM_` prefix stripped) — keep matching filenames on both sides.

Each dataset YAML (`data/m5_*.yaml`) declares:

```yaml
path: <dataset-root>/Images/          # root the train/val/test/benchmark paths below are resolved against
train: 'HCM/train/400x'               # detection + contrastive training split
val:   'LCM/test/400x'                # validation split (cross-domain: LCM)
test:  'LCM/test/400x'

benchmark:                            # optional cross-dataset generalization checks, run once at the end of training
  BBBC041: 'processed_data/BBBC041/test'
  IML: 'processed_data/iml-malaria'

names:
  0: "ring"
  1: "trophozoite"
  2: "schizont"
  3: "gametocyte"
```

Point `data.config` in `config/train_contrastive.yaml` at the YAML for the zoom level you want to train on.

### Cross-dataset benchmarks

At the end of training, the best checkpoint is additionally evaluated (detection metrics only, no contrastive loss) against any datasets listed under `benchmark:` — used here for [BBBC041](https://bbbc.broadinstitute.org/BBBC041) and the IML-malaria dataset, which use their own class taxonomies and are not expected to align 1:1 with the 4 M5 classes.

## Configuration

Everything for a run — data, model, VICRegL, optimizer/schedule, hyperparameters — lives in one YAML, `config/train_contrastive.yaml`. Copy it to start a new experiment instead of passing CLI flags:

| Section    | Key(s)                                             | Meaning                                                                 |
|------------|-----------------------------------------------------|--------------------------------------------------------------------------|
| top-level  | `name`, `out_dir`, `exist_ok`, `device`, `parallel`, `seed`, `resume` | run identity, output location, device string, resume behavior |
| `data`     | `config`, `cache`, `single_cls`, `rect`, `workers`  | dataset YAML path, image caching mode, dataloader options                |
| `model`    | `weights`, `cfg`, `imgsz`, `freeze`, `sync_bn`      | initial weights, architecture YAML, input size, layer freezing           |
| `vicregl`  | `alpha`, `maps_mlp`, `mlp`, `norm_layer`, `loss_weight` | global/local balance, projector head sizes, weight on the total loss |
| `train`    | `epochs`, `batch_size`, `optimizer`, `cos_lr`, `patience`, ... | standard YOLOv5 training options                            |
| `hyp`      | `lr0`, `momentum`, `box`, `cls`, `mosaic`, ...       | YOLOv5 loss/augmentation hyperparameters                                  |
| `evolve`   | `enabled`, `generations`, `bucket`                  | hyperparameter evolution (mutates `hyp` across generations)               |
| `wandb`    | `entity`, `upload_dataset`, `bbox_interval`, `artifact_alias` | optional Weights & Biases logging                             |

## Training

### Single-GPU

```bash
python main.py --config_path config/train_contrastive.yaml
```

Override the device or config path without editing the YAML:

```bash
python main.py --config_path config/train_contrastive.yaml -d 0
```

### Multi-GPU (DDP)

Set `parallel: true` in the config, then launch with `torchrun`, listing every physical GPU you want the job to use in `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 main.py \
    --config_path config/train_contrastive.yaml
```

`train.batch_size` must be divisible by the number of processes, and AutoBatch (`batch_size: -1`) is single-GPU only.

### Resuming a Run

```yaml
resume: true                              # resume the most recent run in <out_dir>
# or
resume: runs/train/exp/weights/last.pt    # resume a specific checkpoint
```

The resumed run adopts the exact config that produced the checkpoint (saved alongside it as `config.yaml`), so hyperparameters stay consistent across the resume.

### Hyperparameter Evolution

```yaml
evolve:
  enabled: true
  generations: 300
```

Each generation runs a full training from scratch with mutated `hyp` values, logging results to `<out_dir>/<name>/evolve.csv` and the best set to `hyp_evolve.yaml`. Not compatible with DDP or `image_weights`.

## Outputs

Each run writes to `<out_dir>/<name>/` (auto-incremented unless `exist_ok: true`):

```
runs/train/<name>/
├── config.yaml              # exact resolved config used for this run (for reproducibility/resume)
├── weights/
│   ├── best.pt               # checkpoint with the best validation fitness
│   └── last.pt
├── results.csv                # per-epoch losses and metrics
├── labels.jpg, labels_correlogram.jpg
├── PR_curve.png, F1_curve.png, P_curve.png, R_curve.png
└── M5/, BBBC041/, IML/        # per-benchmark evaluation artifacts and evaluation_results.txt
                                # written once at the end of training (see Cross-Dataset Benchmarks)
```

## Citation

If you use this repository, please cite MaLDAM alongside the two works it builds on:

```bibtex
@inproceedings{fernando2026maldam,
  title     = {MaLDAM: Masked Localized Domain Adaptation for Malaria Detection in Low-Cost Microscopic Images},
  author    = {Fernando, Muditha and Rathnasekara, Tishan and Nazar, Saeedha and Perera, Avishka and Kaluarachchi, Tharindu},
  booktitle = {HemaRAI},
  year      = {2026}
}

@article{dave2024codamal,
  title   = {CodaMal: Contrastive Domain Adaptation for Malaria Detection in Low-Cost Microscopes},
  author  = {Dave, Ishan Rajendrakumar and de Blegiers, Tristan and Chen, Chen and Shah, Mubarak},
  journal = {arXiv preprint arXiv:2402.10478},
  year    = {2024}
}

@article{bardes2022vicregl,
  title   = {VICRegL: Self-Supervised Learning of Local Visual Features},
  author  = {Bardes, Adrien and Ponce, Jean and LeCun, Yann},
  journal = {arXiv preprint arXiv:2210.01571},
  year    = {2022}
}
```

## Acknowledgements

- Detection backbone and training loop adapted from [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5).
- Contrastive domain-adaptation formulation builds on [CodaMal](https://github.com/intelligentMachines-ITU/LowCostMalariaDetection_CVPR_2022) (Dave et al.) and [VICRegL](https://github.com/facebookresearch/VICRegL) (Bardes et al.).
- M5 dataset courtesy of the CodaMal authors / ITU Intelligent Machines Lab.

## License

Released under the [MIT License](LICENSE).
