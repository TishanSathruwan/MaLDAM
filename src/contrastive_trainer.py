"""
Copyright 2026 MaLDAM Authors. All Rights Reserved.

MaLDAM: Masked Localized Domain Adaptation for Malaria Detection in Low-Cost Microscopic Images.
Authors: Muditha Fernando, Tishan Rathnasekara, Saeedha Nazar, Avishka Perera, Tharindu Kaluarachchi
Publication: HemaRAI 2026 | CC BY 4.0
File: Contrastive Trainer
"""

import math
import os
import random
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import lr_scheduler
from tqdm import tqdm

import src.utils.val as validate
from src.models.experimental import attempt_load
from src.models.yolo import Model
from src.utils.autoanchor import check_anchors
from src.utils.autobatch import check_train_batch_size
from src.utils.dataloaders import create_dataloader as create_val_dataloader
from src.utils.dataloaders_contrastive import create_dataloader
from src.utils.downloads import attempt_download
from src.utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,    
    check_amp,
    check_dataset,
    check_img_size,
    check_suffix,
    check_yaml,
    colorstr,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    strip_optimizer,
)
from src.utils.loggers import Loggers
from src.utils.loss import ComputeLoss
from src.utils.metrics import fitness
from src.utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    smart_DDP,
    smart_optimizer2,
    smart_resume,
    torch_distributed_zero_first,
)
from src.utils.vicregl import VICRegLHead, VICRegLLoss

from src.config import save_config

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


class ContrastiveTrainer:
    """
    VICRegL contrastive pretraining pipeline for YOLOv5, config-driven end to end.

    Args:
        cfg (DictConfig): full run configuration, see config/train_contrastive.yaml.
        device (torch.device): device selected for this process.
        callbacks (Callbacks): YOLOv5 callbacks instance (loggers register onto it).
    """

    def __init__(self, cfg, device, callbacks):
        self.cfg = cfg
        self.device = device
        self.callbacks = callbacks
        self.hyp = OmegaConf.to_container(cfg.hyp, resolve=True)

        self._init_output_dir()
        self.opt = self._build_opt_namespace()
        LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in self.hyp.items()))
        self._save_recipe()

        self._init_loggers()
        self._prepare_dataset()
        self._init_model()
        self._freeze_layers()
        self._resolve_image_size_and_batch()
        self._init_optimizer_and_scheduler()
        self._init_ema()
        self._resume_if_needed()
        self._wrap_data_parallel()
        self._init_dataloaders()
        self._finalize_model_attrs()
        self._init_training_state()

    def _init_output_dir(self):
        """Resolve the run directory"""
        project = self.cfg.out_dir
        if self.cfg.evolve.enabled and project == "runs/train":
            project = "runs/evolve"
        name = Path(self.cfg.model.cfg).stem if self.cfg.name == "cfg" else self.cfg.name
        exist_ok = bool(self.cfg.exist_ok) or bool(self.cfg.evolve.enabled)

        self.save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))
        self.weights_dir = self.save_dir / "weights"
        (self.weights_dir.parent if self.cfg.evolve.enabled else self.weights_dir).mkdir(
            parents=True, exist_ok=True
        )
        self.last = self.weights_dir / "last.pt"
        self.best = self.weights_dir / "best.pt"

    def _build_opt_namespace(self):
        """Flatten cfg into an argparse-style namespac """
        cfg = self.cfg
        evolve = cfg.evolve.generations if cfg.evolve.enabled else None
        return SimpleNamespace(
            weights=str(cfg.model.weights),
            cfg=check_yaml(str(cfg.model.cfg)) if cfg.model.cfg else "",
            data=str(cfg.data.config),
            hyp=dict(self.hyp),
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            imgsz=cfg.model.imgsz,
            rect=cfg.data.rect,
            resume=cfg.resume if isinstance(cfg.resume, str) else bool(cfg.resume),
            nosave=bool(cfg.train.nosave) or bool(cfg.evolve.enabled),
            noval=bool(cfg.train.noval) or bool(cfg.evolve.enabled),
            noautoanchor=cfg.train.noautoanchor,
            noplots=cfg.train.noplots,
            evolve=evolve,
            bucket=cfg.evolve.bucket,
            cache=cfg.data.cache,
            image_weights=cfg.train.image_weights,
            device=str(cfg.device),
            multi_scale=cfg.train.multi_scale,
            single_cls=cfg.data.single_cls,
            optimizer=cfg.train.optimizer,
            sync_bn=cfg.model.sync_bn,
            workers=cfg.data.workers,
            project=str(cfg.out_dir),
            name=str(cfg.name),
            exist_ok=bool(cfg.exist_ok),
            quad=cfg.train.quad,
            cos_lr=cfg.train.cos_lr,
            label_smoothing=cfg.train.label_smoothing,
            patience=cfg.train.patience,
            freeze=list(cfg.model.freeze),
            save_period=cfg.train.save_period,
            seed=cfg.seed,
            local_rank=-1,
            entity=cfg.wandb.entity,
            upload_dataset=cfg.wandb.upload_dataset,
            bbox_interval=cfg.wandb.bbox_interval,
            artifact_alias=cfg.wandb.artifact_alias,
            save_dir=str(self.save_dir),
        )

    def _save_recipe(self):
        """Snapshot the full run configuration once, unless this is an evolution generation."""
        if not self.opt.evolve:
            save_config(self.cfg, self.save_dir / "config.yaml")

    def _init_loggers(self):
        self.data_dict = None
        self.loggers = None
        if RANK in (-1, 0):
            self.loggers = Loggers(self.save_dir, self.opt.weights, self.opt, self.opt.hyp, LOGGER)
            for k in methods(self.loggers):
                self.callbacks.register_action(k, callback=getattr(self.loggers, k))
            self.data_dict = self.loggers.remote_dataset
            if self.opt.resume:
                # opt.* may have been rewritten in-place above (e.g. wandb artifact resume)
                self.hyp = dict(self.opt.hyp) if isinstance(self.opt.hyp, dict) else self.hyp

    def _prepare_dataset(self):
        init_seeds(self.opt.seed + 1 + RANK, deterministic=True)
        with torch_distributed_zero_first(LOCAL_RANK):
            self.data_dict = self.data_dict or check_dataset(self.opt.data)
        self.train_path = self.data_dict["train"]
        self.val_path = self.data_dict["val"]
        benchmark_root = Path(self.data_dict["path"]).parent.parent
        self.bench_paths = {
            bm: str((benchmark_root / p).resolve()) for bm, p in self.data_dict["benchmark"].items()
        }
        self.nc = 1 if self.opt.single_cls else int(self.data_dict["nc"])
        self.names = (
            {0: "item"}
            if self.opt.single_cls and len(self.data_dict["names"]) != 1
            else self.data_dict["names"]
        )
        self.is_coco = isinstance(self.val_path, str) and self.val_path.endswith("coco/val2017.txt")
        self.plots = not self.opt.evolve and not self.opt.noplots
        self.cuda = self.device.type != "cpu"

    def _init_model(self):
        check_suffix(self.opt.weights, ".pt")
        self.pretrained = self.opt.weights.endswith(".pt")
        self._ckpt = None
        self._csd = None

        if self.pretrained:
            with torch_distributed_zero_first(LOCAL_RANK):
                weights = attempt_download(self.opt.weights)
            self._ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            self.model = Model(
                self.opt.cfg or self._ckpt["model"].yaml, ch=3, nc=self.nc, anchors=self.hyp.get("anchors")
            ).to(self.device)
            exclude = ["anchor"] if (self.opt.cfg or self.hyp.get("anchors")) and not self.opt.resume else []
            self._csd = self._ckpt["model"].float().state_dict()
            self._csd = intersect_dicts(self._csd, self.model.state_dict(), exclude=exclude)
            self.model.load_state_dict(self._csd, strict=False)
            LOGGER.info(f"Transferred {len(self._csd)}/{len(self.model.state_dict())} items from {weights}")
        else:
            self.model = Model(self.opt.cfg, ch=3, nc=self.nc, anchors=self.hyp.get("anchors")).to(self.device)

        self.projection_head = VICRegLHead(
            alpha=self.cfg.vicregl.alpha,
            maps_mlp=self.cfg.vicregl.maps_mlp,
            mlp=self.cfg.vicregl.mlp,
            norm_layer=self.cfg.vicregl.norm_layer,
        ).to(self.device)

        self.amp = check_amp(self.model)

    def _freeze_layers(self):
        freeze = self.opt.freeze
        freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
        for k, v in self.model.named_parameters():
            v.requires_grad = True
            if any(x in k for x in freeze):
                LOGGER.info(f"freezing {k}")
                v.requires_grad = False

    def _resolve_image_size_and_batch(self):
        self.gs = max(int(self.model.stride.max()), 32)
        self.imgsz = check_img_size(self.opt.imgsz, self.gs, floor=self.gs * 2)
        self.batch_size = self.opt.batch_size
        if RANK == -1 and self.batch_size == -1:
            self.batch_size = check_train_batch_size(self.model, self.imgsz, self.amp) / 3  # 3x for contrastive
            self.loggers.on_params_update({"batch_size": self.batch_size})

    def _init_optimizer_and_scheduler(self):
        self.nbs = 64  # nominal batch size
        self.accumulate = max(round(self.nbs / self.batch_size), 1)
        self.hyp["weight_decay"] *= self.batch_size * self.accumulate / self.nbs
        self.optimizer = smart_optimizer2(
            self.model,
            self.projection_head,
            self.opt.optimizer,
            self.hyp["lr0"],
            self.hyp["momentum"],
            self.hyp["weight_decay"],
        )

        if self.opt.cos_lr:
            self.lf = one_cycle(1, self.hyp["lrf"], self.opt.epochs)
        else:
            self.lf = lambda x: (1 - x / self.opt.epochs) * (1.0 - self.hyp["lrf"]) + self.hyp["lrf"]
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _init_ema(self):
        self.ema = ModelEMA(self.model) if RANK in (-1, 0) else None

    def _resume_if_needed(self):
        self.best_fitness, self.start_epoch = 0.0, 0
        self.epochs = self.opt.epochs
        if self.pretrained:
            if self.opt.resume:
                self.best_fitness, self.start_epoch, self.epochs = smart_resume(
                    self._ckpt, self.optimizer, self.ema, self.opt.weights, self.epochs, self.opt.resume
                )
            del self._ckpt, self._csd

    def _wrap_data_parallel(self):
        if self.cuda and RANK == -1 and torch.cuda.device_count() > 1:
            LOGGER.warning(
                "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
                "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training "
                "to get started."
            )
            self.model = torch.nn.DataParallel(self.model)
            self.projection_head = torch.nn.DataParallel(self.projection_head)

        if self.opt.sync_bn and self.cuda and RANK != -1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            self.projection_head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.projection_head).to(
                self.device
            )
            LOGGER.info("Using SyncBatchNorm()")

    def _init_dataloaders(self):
        self.train_loader, self.dataset = create_dataloader(
            self.train_path,
            self.imgsz,
            self.batch_size // WORLD_SIZE,
            self.gs,
            self.opt.single_cls,
            hyp=self.hyp,
            augment=True,
            dg_augment=True,
            cache=None if self.opt.cache == "val" else self.opt.cache,
            rect=self.opt.rect,
            rank=LOCAL_RANK,
            workers=self.opt.workers,
            image_weights=self.opt.image_weights,
            quad=self.opt.quad,
            prefix=colorstr("train: "),
            shuffle=True,
            seed=self.opt.seed,
        )
        labels = np.concatenate(self.dataset.labels, 0)
        mlc = int(labels[:, 0].max())
        assert mlc < self.nc, (
            f"Label class {mlc} exceeds nc={self.nc} in {self.opt.data}. "
            f"Possible class labels are 0-{self.nc - 1}"
        )

        self.val_loader = None
        self.bench_loaders = {}
        if RANK in (-1, 0):
            self.val_loader = create_val_dataloader(
                self.val_path,
                self.imgsz,
                self.batch_size // WORLD_SIZE * 2,
                self.gs,
                self.opt.single_cls,
                hyp=self.hyp,
                cache=None if self.opt.noval else self.opt.cache,
                rect=True,
                rank=-1,
                workers=self.opt.workers * 2,
                pad=0.5,
                prefix=colorstr("val: "),
            )[0]

            self.bench_loaders = {
                bm: create_val_dataloader(
                    path,
                    self.imgsz,
                    self.batch_size // WORLD_SIZE * 2,
                    self.gs,
                    self.opt.single_cls,
                    hyp=self.hyp,
                    cache=None if self.opt.noval else self.opt.cache,
                    rect=True,
                    rank=-1,
                    workers=self.opt.workers * 2,
                    pad=0.5,
                    prefix=colorstr(f"bench-{bm}: "),
                )[0]
                for bm, path in self.bench_paths.items()
            }

            if not self.opt.resume:
                if not self.opt.noautoanchor:
                    check_anchors(self.dataset, model=self.model, thr=self.hyp["anchor_t"], imgsz=self.imgsz)
                self.model.half().float()

            self.callbacks.run("on_pretrain_routine_end", labels, self.names)

        if self.cuda and RANK != -1:
            self.model = smart_DDP(self.model)
            self.projection_head = smart_DDP(self.projection_head)

    def _finalize_model_attrs(self):
        nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        self.hyp["box"] *= 3 / nl
        self.hyp["cls"] *= self.nc / 80 * 3 / nl
        self.hyp["obj"] *= (self.imgsz / 640) ** 2 * 3 / nl
        self.hyp["label_smoothing"] = self.opt.label_smoothing
        self.model.nc = self.nc
        self.model.hyp = self.hyp
        self.model.class_weights = labels_to_class_weights(self.dataset.labels, self.nc).to(self.device) * self.nc
        self.model.names = self.names

    def _init_training_state(self):
        self.nb = len(self.train_loader)
        self.nw = max(round(self.hyp["warmup_epochs"] * self.nb), 100)
        self.last_opt_step = -1
        self.maps = np.zeros(self.nc)
        self.results = (0, 0, 0, 0, 0, 0, 0)
        self.scheduler.last_epoch = self.start_epoch - 1
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.stopper = EarlyStopping(patience=self.opt.patience)
        self.stop = False
        self.compute_loss = ComputeLoss(self.model)
        self.vicregl_criterion = VICRegLLoss(alpha=self.cfg.vicregl.alpha)
        self._last_mloss, self._last_lr, self._last_fi = [0.0, 0.0, 0.0], [0.0], 0.0

        self.t0 = time.time()
        self.callbacks.run("on_train_start")
        LOGGER.info(
            f"Image sizes {self.imgsz} train, {self.imgsz} val\n"
            f"Using {self.train_loader.num_workers * WORLD_SIZE} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for {self.epochs} epochs..."
        )

    # -------------------------------------------------------------- training

    def _apply_image_weights(self):
        if self.opt.image_weights:
            cw = self.model.class_weights.cpu().numpy() * (1 - self.maps) ** 2 / self.nc
            iw = labels_to_image_weights(self.dataset.labels, nc=self.nc, class_weights=cw)
            self.dataset.indices = random.choices(range(self.dataset.n), weights=iw, k=self.dataset.n)

    def _train_one_epoch(self, epoch):
        """Train for one epoch. Returns (mloss, lr) or (None, None) if callbacks requested a stop."""
        self.callbacks.run("on_train_epoch_start")
        self.model.train()
        self.projection_head.train()
        self._apply_image_weights()

        mloss = torch.zeros(3, device=self.device)
        mdac_loss = 0.0
        if RANK != -1:
            self.train_loader.sampler.set_epoch(epoch)

        pbar = enumerate(self.train_loader)
        LOGGER.info(
            ("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "Instances", "Size")
        )
        if RANK in (-1, 0):
            pbar = tqdm(pbar, total=self.nb, bar_format=TQDM_BAR_FORMAT)
        self.optimizer.zero_grad()

        for i, (imgs1, targets, paths, _, mh, ml) in pbar:
            imgs, imgs_hcm, imgs_lcm = imgs1
            mh = mh.flatten(1)
            ml = ml.flatten(1)

            self.callbacks.run("on_train_batch_start")
            ni = i + self.nb * epoch  # number integrated batches (since train start)

            imgs = imgs.to(self.device, non_blocking=True).float() / 255
            imgs_hcm = imgs_hcm.to(self.device, non_blocking=True).float() / 255
            imgs_lcm = imgs_lcm.to(self.device, non_blocking=True).float() / 255

            # Warmup
            if ni <= self.nw:
                xi = [0, self.nw]
                self.accumulate = max(1, np.interp(ni, xi, [1, self.nbs / self.batch_size]).round())
                for j, x in enumerate(self.optimizer.param_groups):
                    x["lr"] = np.interp(
                        ni,
                        xi,
                        [self.hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)],
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [self.hyp["warmup_momentum"], self.hyp["momentum"]])

            # Multi-scale
            if self.opt.multi_scale:
                sz = random.randrange(int(self.imgsz * 0.5), int(self.imgsz * 1.5) + self.gs) // self.gs * self.gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
                    imgs_hcm = nn.functional.interpolate(imgs_hcm, size=ns, mode="bilinear", align_corners=False)
                    imgs_lcm = nn.functional.interpolate(imgs_lcm, size=ns, mode="bilinear", align_corners=False)

            # Forward
            with torch.amp.autocast(device_type="cuda", enabled=self.amp):
                pred = self.model(imgs)
                _, pred_hcm = self.model(imgs_hcm, features_output=True)
                _, pred_lcm = self.model(imgs_lcm, features_output=True)

                # vicregL based features
                vicregl_outs = self.projection_head([pred_hcm, pred_lcm])

                # extracting FG mask for masked local matching
                fg_mh = mh > 0
                fg_lh = ml > 0
                fg_mask = fg_mh.unsqueeze(2) & fg_lh.unsqueeze(1)
                fg_mask = ~fg_mask

                loss, loss_items = self.compute_loss(pred, targets.to(self.device))
                vicregl_loss, _ = self.vicregl_criterion(vicregl_outs, fg_mask)
                vicregl_loss = vicregl_loss * self.cfg.vicregl.loss_weight

                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if self.opt.quad:
                    loss *= 4.0

            # Backward
            self.scaler.scale(loss).backward()
            self.scaler.scale(vicregl_loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - self.last_opt_step >= self.accumulate:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.ema:
                    self.ema.update(self.model)
                self.last_opt_step = ni

            # Log
            if RANK in (-1, 0):
                mloss = (mloss * i + loss_items) / (i + 1)
                mdac_loss = (mdac_loss * i + vicregl_loss.detach().item()) / (i + 1)
                mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"
                pbar.set_description(
                    ("%11s" * 3 + "%11.4g" * 6)
                    % (
                        f"{epoch}/{self.epochs - 1}",
                        mem,
                        f"DAC {mdac_loss:.4g}",
                        *mloss,
                        mdac_loss,
                        targets.shape[0],
                        imgs.shape[-1],
                    )
                )
                self.callbacks.run(
                    "on_train_batch_end", self.model, ni, imgs, targets, paths, list(mloss) + [mdac_loss]
                )
                if self.callbacks.stop_training:
                    return None, None
            # end batch ------------------------------------------------------------------------------------

        lr = [x["lr"] for x in self.optimizer.param_groups]
        self.scheduler.step()
        return list(mloss), lr

    def _validate(self, epoch, final_epoch):
        if RANK not in (-1, 0):
            return self.results, self.maps

        self.callbacks.run("on_train_epoch_end", epoch=epoch)
        self.ema.update_attr(self.model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
        if not self.opt.noval or final_epoch:  # Calculate mAP
            self.results, self.maps, _ = validate.run(
                self.data_dict,
                batch_size=self.batch_size // WORLD_SIZE * 2,
                imgsz=self.imgsz,
                half=self.amp,
                model=self.ema.ema,
                single_cls=self.opt.single_cls,
                dataloader=self.val_loader,
                save_dir=self.save_dir,
                plots=False,
                callbacks=self.callbacks,
                compute_loss=self.compute_loss,
            )
        return self.results, self.maps

    def _save_checkpoint(self, epoch, fitness_value, final_epoch):
        ckpt = {
            "epoch": epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(de_parallel(self.model)).half(),
            "projection_head": deepcopy(de_parallel(self.projection_head)).half(),
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(),
            "opt": vars(self.opt),
            "date": datetime.now().isoformat(),
        }

        torch.save(ckpt, self.last)
        if self.best_fitness == fitness_value:
            torch.save(ckpt, self.best)
        if self.opt.save_period > 0 and epoch % self.opt.save_period == 0:
            torch.save(ckpt, self.weights_dir / f"epoch{epoch}.pt")
        del ckpt
        self.callbacks.run("on_model_save", self.last, epoch, final_epoch, self.best_fitness, fitness_value)

    def fit(self):
        """Run the full training loop and return the final validation/benchmark results."""
        epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.epochs):
            mloss, lr = self._train_one_epoch(epoch)
            if mloss is None:
                return self.results

            final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

            if RANK in (-1, 0):
                self.results, self.maps = self._validate(epoch, final_epoch)
                fi = fitness(np.array(self.results).reshape(1, -1))
                self.stop = self.stopper(epoch=epoch, fitness=fi)
                if fi > self.best_fitness:
                    self.best_fitness = fi
                log_vals = mloss + list(self.results) + lr
                self.callbacks.run("on_fit_epoch_end", log_vals, epoch, self.best_fitness, fi)
                self._last_mloss, self._last_lr, self._last_fi = mloss, lr, fi

                if (not self.opt.nosave) or (final_epoch and not self.opt.evolve):
                    self._save_checkpoint(epoch, fi, final_epoch)

            # EarlyStopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        self._finalize(epoch)
        return self.results

    def _finalize(self, epoch):
        if RANK in (-1, 0):
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {(time.time() - self.t0) / 3600:.3f} hours.")
            for f in (self.last, self.best):
                if not f.exists():
                    continue
                strip_optimizer(f)
                if f is not self.best:
                    continue

                LOGGER.info(f"\nBenchmarking {f}...")
                bench_dict = {}

                LOGGER.info("\nM5 Testset Benchmarking...")
                m5_save_dir = self.save_dir / "M5"
                m5_save_dir.mkdir(parents=True, exist_ok=True)
                self.results, _, _ = validate.run(
                    self.data_dict,
                    batch_size=self.batch_size // WORLD_SIZE,
                    imgsz=self.imgsz,
                    model=attempt_load(f, self.device).half(),
                    iou_thres=0.65 if self.is_coco else 0.60,
                    single_cls=self.opt.single_cls,
                    dataloader=self.val_loader,
                    save_dir=m5_save_dir,
                    save_json=self.is_coco,
                    verbose=True,
                    plots=self.plots,
                    callbacks=self.callbacks,
                    compute_loss=self.compute_loss,
                    save_class_stats=True,
                )
                bench_dict["M5"] = self.results

                for bench, bench_dl in self.bench_loaders.items():
                    LOGGER.info(f"\n{bench} Benchmarking...")
                    bench_save_dir = self.save_dir / bench
                    bench_save_dir.mkdir(parents=True, exist_ok=True)
                    bench_result, _, _ = validate.run(
                        self.data_dict,
                        batch_size=self.batch_size // WORLD_SIZE,
                        imgsz=self.imgsz,
                        model=attempt_load(f, self.device).half(),
                        iou_thres=0.65 if self.is_coco else 0.60,
                        single_cls=self.opt.single_cls,
                        dataloader=bench_dl,
                        save_dir=bench_save_dir,
                        save_json=self.is_coco,
                        verbose=True,
                        plots=self.plots,
                        callbacks=self.callbacks,
                        # Benchmark sets (e.g. BBBC041) use a different class taxonomy than
                        # the trained model, so class ids can exceed self.compute_loss's nc
                        # and crash the CUDA cls-loss indexing. Detection metrics don't need it.
                        compute_loss=None,
                        save_class_stats=True,
                    )
                    bench_dict[bench] = bench_result

                if self.is_coco:
                    self.callbacks.run(
                        "on_fit_epoch_end",
                        self._last_mloss + list(self.results) + self._last_lr,
                        epoch,
                        self.best_fitness,
                        self._last_fi,
                    )

            self.callbacks.run("on_train_end", self.last, self.best, epoch, self.results)

        torch.cuda.empty_cache()
