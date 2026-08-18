"""
Copyright 2026 MaLDAM Authors. All Rights Reserved.

MaLDAM: Masked Localized Domain Adaptation for Malaria Detection in Low-Cost Microscopic Images.
Authors: Muditha Fernando, Tishan Rathnasekara, Saeedha Nazar, Avishka Perera, Tharindu Kaluarachchi
File: Main training pipeline for VICRegL contrastive pretraining of YOLOv5.
"""

import argparse
import copy
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from src.config import load_config
from src.contrastive_trainer import LOCAL_RANK, RANK, WORLD_SIZE, ContrastiveTrainer
from src.utils.callbacks import Callbacks
from src.utils.general import (
    LOGGER,
    check_git_status,
    check_requirements,
    colorstr,
    get_latest_run,
    print_args,
    print_mutation,
)
from src.utils.metrics import fitness
from src.utils.plots import plot_evolve
from src.utils.torch_utils import select_device

# Hyperparameter evolution metadata
EVOLVE_META = {
    "lr0": (1, 1e-5, 1e-1),
    "lrf": (1, 0.01, 1.0),
    "momentum": (0.3, 0.6, 0.98),
    "weight_decay": (1, 0.0, 0.001),
    "warmup_epochs": (1, 0.0, 5.0),
    "warmup_momentum": (1, 0.0, 0.95),
    "warmup_bias_lr": (1, 0.0, 0.2),
    "box": (1, 0.02, 0.2),
    "cls": (1, 0.2, 4.0),
    "cls_pw": (1, 0.5, 2.0),
    "obj": (1, 0.2, 4.0),
    "obj_pw": (1, 0.5, 2.0),
    "iou_t": (0, 0.1, 0.7),
    "anchor_t": (1, 2.0, 8.0),
    "fl_gamma": (0, 0.0, 2.0),
    "hsv_h": (1, 0.0, 0.1),
    "hsv_s": (1, 0.0, 0.9),
    "hsv_v": (1, 0.0, 0.9),
    "degrees": (1, 0.0, 45.0),
    "translate": (1, 0.0, 0.9),
    "scale": (1, 0.0, 0.9),
    "shear": (1, 0.0, 10.0),
    "perspective": (0, 0.0, 0.001),
    "flipud": (1, 0.0, 1.0),
    "fliplr": (0, 0.0, 1.0),
    "mosaic": (1, 0.0, 1.0),
    "mixup": (1, 0.0, 1.0),
    "copy_paste": (1, 0.0, 1.0),
}


def _resolve_resume(cfg):
    """Adopt the previous run's own saved config."""
    if not cfg.resume or cfg.evolve.enabled:
        return cfg

    last = Path(cfg.resume) if isinstance(cfg.resume, str) else Path(get_latest_run())
    saved_cfg_path = last.parent.parent / "config.yaml"
    if not saved_cfg_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume: no config.yaml found next to {last} (expected at {saved_cfg_path})."
        )

    resumed_cfg = load_config(str(saved_cfg_path))
    resumed_cfg.model.weights = str(last)
    resumed_cfg.resume = True
    resumed_cfg.exist_ok = True
    return resumed_cfg


def run_evolve(cfg, device):
    """Hyperparameter evolution, each a full training run from scratch."""
    hyp = dict(cfg.hyp)
    project = "runs/evolve" if cfg.out_dir == "runs/train" else cfg.out_dir
    save_dir = Path(project) / cfg.name
    save_dir.mkdir(parents=True, exist_ok=True)
    evolve_csv = save_dir / "evolve.csv"
    evolve_yaml = save_dir / "hyp_evolve.yaml"
    bucket = cfg.evolve.bucket

    if bucket:
        subprocess.run(["gsutil", "cp", f"gs://{bucket}/evolve.csv", str(evolve_csv)])

    for _ in range(cfg.evolve.generations):
        if evolve_csv.exists():  # select best hyps and mutate
            parent = "single"  # parent selection method: 'single' or 'weighted'
            x = np.loadtxt(evolve_csv, ndmin=2, delimiter=",", skiprows=1)
            n = min(5, len(x))  # number of previous results to consider
            x = x[np.argsort(-fitness(x))][:n]  # top n mutations
            w = fitness(x) - fitness(x).min() + 1e-6  # weights (sum > 0)
            if parent == "single" or len(x) == 1:
                x = x[random.choices(range(n), weights=w)[0]]
            elif parent == "weighted":
                x = (x * w.reshape(n, 1)).sum(0) / w.sum()

            # Mutate
            mp, s = 0.8, 0.2  # mutation probability, sigma
            npr = np.random
            npr.seed(int(time.time()))
            g = np.array([EVOLVE_META[k][0] for k in hyp.keys()])
            ng = len(hyp)
            v = np.ones(ng)
            while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
            for i, k in enumerate(hyp.keys()):
                hyp[k] = float(x[i + 7] * v[i])

        # Constrain to limits
        for k in hyp:
            hyp[k] = max(hyp[k], EVOLVE_META[k][1])
            hyp[k] = min(hyp[k], EVOLVE_META[k][2])
            hyp[k] = round(hyp[k], 5)

        gen_cfg = copy.deepcopy(cfg)
        gen_cfg.hyp = hyp
        trainer = ContrastiveTrainer(gen_cfg, device, Callbacks())
        results = trainer.fit()

        keys = (
            "metrics/precision",
            "metrics/recall",
            "metrics/mAP_0.5",
            "metrics/mAP_0.5:0.95",
            "val/box_loss",
            "val/obj_loss",
            "val/cls_loss",
        )
        print_mutation(keys, results, hyp.copy(), save_dir, bucket)

    plot_evolve(evolve_csv)
    LOGGER.info(
        f"Hyperparameter evolution finished {cfg.evolve.generations} generations\n"
        f"Results saved to {colorstr('bold', save_dir)}\n"
        f"Usage example: $ python train_main.py --config_path {evolve_yaml}"
    )


def main(cfg):
    """ Main entry point for training."""
    if RANK in (-1, 0):
        print_args(OmegaConf.to_container(cfg, resolve=True))
        check_git_status()
        check_requirements()

    assert len(cfg.model.cfg) or len(cfg.model.weights), "either model.cfg or model.weights must be specified"

    cfg = _resolve_resume(cfg)
    device = select_device('' if LOCAL_RANK != -1 else str(cfg.device), batch_size=cfg.train.batch_size)
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"
        assert not cfg.train.image_weights, f"train.image_weights {msg}"
        assert not cfg.evolve.enabled, f"evolve.enabled {msg}"
        assert cfg.train.batch_size != -1, f"AutoBatch with train.batch_size -1 {msg}, please set a fixed batch size"
        assert cfg.train.batch_size % WORLD_SIZE == 0, f"train.batch_size {cfg.train.batch_size} must be a multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    if cfg.evolve.enabled:
        run_evolve(cfg, device)
    else:
        trainer = ContrastiveTrainer(cfg, device, Callbacks())
        trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaLDAM Training")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/train_contrastive.yaml",
        help="Path to the training configuration YAML",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Override cfg.device, e.g. '0', '0,1', or 'cpu'",
    )
    args = parser.parse_args()

    CFG = load_config(args.config_path)
    if args.device is not None:
        CFG.device = args.device

    main(CFG)
    
# How to train;
# Usage - Single-GPU training:
#     $ python main.py --config_path config/train_contrastive.yaml
# 
# Usage - Multi-GPU DDP training:
#     $ CUDA_VISIBLE_DEVICES=0,3 torchrun --standalone --nproc_per_node=2 main.py \
#         --config_path config/train_contrastive.yaml
