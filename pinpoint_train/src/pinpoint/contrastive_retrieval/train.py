# Train the dual-image contrastive geolocation model with a shared image trunk and source adapters.
# Usage: PYTHONPATH=src python src/pinpoint/contrastive_retrieval/train.py --train-datasets mp16,osv5m --val-datasets mp16,osv5m
"""
Train a dual-image geolocation alignment model with PyTorch Lightning.

Streams cached image embeddings and geographic coordinates from memory-mapped binaries.
Logs to TensorBoard.
"""

from __future__ import annotations

import argparse
from typing import Any

import lightning as L
import numpy as np
import torch

torch.set_float32_matmul_precision("high")

from pinpoint.contrastive_retrieval.lightning_module import GeoLightningModule
from pinpoint.dataset.embed_data import add_embed_data_args
from pinpoint.dataset.metadata_datamodule import build_metadata_datamodule_and_dim
from pinpoint.utils import (
    PeriodicCheckpointCallback,
    create_default_tb_logger,
    create_default_trainer,
    log_run_config_to_tensorboard,
    prepare_experiment_dirs,
    resolve_resume_training_state,
    trainer_device_kwargs,
)


def _pred_to_deg(_pl_module, pred_tensor):
    return pred_tensor


def _parse_comma_separated_numbers(raw: str | None, cast, name: str) -> list[int] | list[float] | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return []
    values = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            raise ValueError(f"{name} must not contain empty items")
        values.append(cast(item))
    return values


def main() -> None:
    """Train the conference submission's contrastive retrieval backbone."""
    parser = argparse.ArgumentParser(
        description="Train contrastive geolocation with a shared image trunk and MP16/OSV adapters"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Number of linear warmup steps before cosine decay; 0 disables warmup.",
    )
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--val-batches", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=20,
        help="Retain only the most recent N ckpt_step*.pt checkpoints; 0 disables pruning.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="experiments",
        help="Root folder for run outputs (checkpoints, logs, tfevents).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Run folder name under --experiment-dir",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=6,
        help="DataLoader prefetch_factor when num_workers > 0.",
    )
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--device", type=str, default=None)
    add_embed_data_args(parser)
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--shared-image-hidden-dim", type=int, default=1024)
    parser.add_argument("--shared-image-depth", type=int, default=3)
    parser.add_argument("--shared-image-expansion", type=int, default=2)
    parser.add_argument("--shared-image-dropout", type=float, default=0.15)
    parser.add_argument("--image-adapter-depth", type=int, default=1)
    parser.add_argument("--image-adapter-expansion", type=int, default=2)
    parser.add_argument("--image-adapter-dropout", type=float, default=0.15)
    parser.add_argument("--loc-hidden-dim", type=int, default=1024)
    parser.add_argument("--loc-depth", type=int, default=3)
    parser.add_argument("--loc-expansion", type=int, default=2)
    parser.add_argument("--loc-dropout", type=float, default=0.10)
    parser.add_argument(
        "--use-rff",
        action="store_true",
        help="Enable random Fourier features in the location tower. Disabled by default.",
    )
    parser.add_argument(
        "--use-learned-rff",
        action="store_true",
        help="Enable trainable Fourier features in the location tower. Disabled by default.",
    )
    parser.add_argument("--rff-seed", type=int, default=21)
    parser.add_argument(
        "--rff-features-list",
        type=str,
        default=None,
        help="Optional comma-separated Fourier feature counts for one or more location feature blocks.",
    )
    parser.add_argument(
        "--rff-sigmas-list",
        type=str,
        default=None,
        help="Optional comma-separated sigmas for one or more location feature blocks.",
    )
    parser.add_argument("--logit-scale-init", type=float, default=10.0)
    parser.add_argument("--logit-scale-max", type=float, default=100.0)
    parser.add_argument("--logit-bias-init", type=float, default=-10.0)
    parser.add_argument("--loss-weight-image-loc", type=float, default=1.0)
    parser.add_argument("--loss-weight-osv-image-loc", type=float, default=1.0)
    parser.set_defaults(train_mixing="uniform")
    args = parser.parse_args()
    args.rff_features_list = _parse_comma_separated_numbers(args.rff_features_list, int, "--rff-features-list")
    args.rff_sigmas_list = _parse_comma_separated_numbers(args.rff_sigmas_list, float, "--rff-sigmas-list")

    if args.keep_last_checkpoints < 0:
        raise ValueError("--keep-last-checkpoints must be >= 0")
    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.warmup_steps >= args.steps:
        raise ValueError("--warmup-steps must be < --steps")
    if args.val_every <= 0:
        raise ValueError("--val-every must be > 0")
    if args.val_batches <= 0:
        raise ValueError("--val-batches must be > 0")
    if args.log_every <= 0:
        raise ValueError("--log-every must be > 0")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be > 0")
    if not (0 <= args.val_pct < 100):
        raise ValueError("--val-pct must be in [0, 99]")
    if args.output_dim <= 0:
        raise ValueError("--output-dim must be > 0")
    if args.shared_image_hidden_dim <= 0:
        raise ValueError("--shared-image-hidden-dim must be > 0")
    if args.shared_image_depth <= 0:
        raise ValueError("--shared-image-depth must be > 0")
    if args.shared_image_expansion <= 0:
        raise ValueError("--shared-image-expansion must be > 0")
    if args.image_adapter_depth <= 0:
        raise ValueError("--image-adapter-depth must be > 0")
    if args.image_adapter_expansion <= 0:
        raise ValueError("--image-adapter-expansion must be > 0")
    if args.use_rff and args.use_learned_rff:
        raise ValueError("--use-rff and --use-learned-rff are mutually exclusive")
    if (args.rff_features_list is None) != (args.rff_sigmas_list is None):
        raise ValueError("--rff-features-list and --rff-sigmas-list must be provided together")
    if args.rff_features_list is not None:
        if len(args.rff_features_list) != len(args.rff_sigmas_list):
            raise ValueError("--rff-features-list and --rff-sigmas-list must have matching lengths")
        if len(args.rff_features_list) == 0:
            raise ValueError("--rff-features-list and --rff-sigmas-list must not be empty")
        if any(v <= 0 for v in args.rff_features_list):
            raise ValueError("--rff-features-list entries must all be > 0")
        if any(v <= 0 for v in args.rff_sigmas_list):
            raise ValueError("--rff-sigmas-list entries must all be > 0")
        if not (args.use_rff or args.use_learned_rff):
            raise ValueError("--rff-features-list requires --use-rff or --use-learned-rff")
    else:
        if args.use_rff or args.use_learned_rff:
            raise ValueError(
                "List-based Fourier settings are required when --use-rff or --use-learned-rff is enabled"
            )
    if args.logit_scale_init <= 0:
        raise ValueError("--logit-scale-init must be > 0")
    if args.logit_scale_max <= 0:
        raise ValueError("--logit-scale-max must be > 0")
    if args.loss_weight_image_loc < 0:
        raise ValueError("--loss-weight-image-loc must be >= 0")
    if args.loss_weight_osv_image_loc < 0:
        raise ValueError("--loss-weight-osv-image-loc must be >= 0")
    if args.loss_weight_image_loc == 0 and args.loss_weight_osv_image_loc == 0:
        raise ValueError("At least one location alignment loss weight must be > 0")

    L.seed_everything(args.seed, workers=True)
    np.random.seed(args.seed)

    resume_path, fit_ckpt_path = resolve_resume_training_state(args)
    run_dir, _, _ = prepare_experiment_dirs(args, resume_path)

    device_kwargs = trainer_device_kwargs(args.device)
    print(f"Trainer device config: {device_kwargs}")

    print("Loading embedding cache metadata...")
    datamodule, embed_dim, train_datasets, val_datasets = build_metadata_datamodule_and_dim(
        args,
        default_train_mixing="uniform",
    )
    print(f"Train datasets: {','.join(train_datasets)}")
    print(f"Val datasets: {','.join(val_datasets)}")
    if len(train_datasets) > 1:
        print(f"Train mixing: {args.train_mixing}")

    selected_datasets = set(train_datasets + val_datasets)
    if "mp16" in selected_datasets and args.loss_weight_image_loc <= 0:
        raise ValueError("--loss-weight-image-loc must be > 0 when mp16 is selected")
    if "osv5m" in selected_datasets and args.loss_weight_osv_image_loc <= 0:
        raise ValueError("--loss-weight-osv-image-loc must be > 0 when osv5m is selected")

    print(f"Input embedding dim: {embed_dim}")

    module = GeoLightningModule(
        embed_dim=embed_dim,
        image_embed_dim=embed_dim,
        osv_image_embed_dim=embed_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        output_dim=args.output_dim,
        shared_image_hidden_dim=args.shared_image_hidden_dim,
        shared_image_depth=args.shared_image_depth,
        shared_image_expansion=args.shared_image_expansion,
        shared_image_dropout=args.shared_image_dropout,
        image_adapter_depth=args.image_adapter_depth,
        image_adapter_expansion=args.image_adapter_expansion,
        image_adapter_dropout=args.image_adapter_dropout,
        use_osv_image_tower=("osv5m" in selected_datasets),
        loc_hidden_dim=args.loc_hidden_dim,
        loc_depth=args.loc_depth,
        loc_expansion=args.loc_expansion,
        loc_dropout=args.loc_dropout,
        use_rff=args.use_rff,
        use_learned_rff=args.use_learned_rff,
        rff_seed=args.rff_seed,
        rff_feature_sizes=args.rff_features_list,
        rff_sigmas=args.rff_sigmas_list,
        logit_scale_init=args.logit_scale_init,
        logit_scale_max=args.logit_scale_max,
        logit_bias_init=args.logit_bias_init,
        loss_weight_image_loc=args.loss_weight_image_loc,
        loss_weight_osv_image_loc=args.loss_weight_osv_image_loc,
    )
    print(f"Embedding dim: {module.embed_dim}")
    print(f"OSV image tower enabled: {module.has_osv_image_tower}")
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Dual-image trainable parameters: {trainable_params:,}")

    tb_logger = create_default_tb_logger(run_dir)

    checkpoint_cb = PeriodicCheckpointCallback(
        checkpoint_dir=args.checkpoint_dir,
        keep_last=args.keep_last_checkpoints,
        pred_to_deg_fn=_pred_to_deg,
    )
    trainer = create_default_trainer(
        args,
        logger=tb_logger,
        callbacks=[checkpoint_cb],
        device_kwargs=device_kwargs,
    )
    log_run_config_to_tensorboard(trainer, tb_logger, args)
    print(f"Training for {args.steps} steps...")
    trainer.fit(module, datamodule=datamodule, ckpt_path=fit_ckpt_path)
    print("Training complete.")


if __name__ == "__main__":
    main()
