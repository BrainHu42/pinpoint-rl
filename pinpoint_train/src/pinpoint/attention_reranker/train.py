# Train the top-N attention reranker over a cached multimodal retrieval index.
# Usage: PYTHONPATH=src python src/pinpoint/attention_reranker/train.py --contrastive-checkpoint-path /path/to/contrastive.ckpt --train-datasets mp16 --val-datasets mp16
"""Train a top-N geolocation reranker over a cached multimodal retrieval index."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import numpy as np
import torch

torch.set_float32_matmul_precision("high")

from pinpoint.attention_reranker.lightning_module import AttentionInterpolateLightningModule
from pinpoint.attention_reranker.retrieval_index import ensure_attention_retrieval_index
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


def main() -> None:
    """Train the conference submission's attention reranker from cached embeddings."""
    parser = argparse.ArgumentParser(description="Train top-N geolocation reranker over retrieval candidates")
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
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda",),
        help="Training runs on a CUDA GPU with bfloat16 support.",
    )
    add_embed_data_args(parser)

    parser.add_argument(
        "--contrastive-checkpoint-path",
        type=str,
        required=True,
        help="Path to the trained contrastive_retrieval checkpoint used for query/image-gps projection.",
    )
    parser.add_argument(
        "--retrieval-index-dir",
        type=str,
        default=None,
        help=(
            "Path to retrieval index folder (or base folder used to cache built indices). "
            "When unset, defaults to <contrastive_ckpt_dir>/attention_retrieval_index."
        ),
    )
    parser.add_argument(
        "--retrieval-split",
        choices=("all", "train", "val"),
        default="train",
        help="Which cache split to index for retrieval.",
    )
    parser.add_argument("--retrieval-build-batch-size", type=int, default=8192)
    parser.add_argument(
        "--retrieval-build-device",
        type=str,
        default="cuda",
        choices=("cuda",),
        help="Device used for retrieval index build projection.",
    )
    parser.add_argument(
        "--retrieval-build-dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float32"),
        help="Autocast dtype used during retrieval index build. Use float32 only when extra precision is needed.",
    )
    parser.add_argument(
        "--retrieval-db-source",
        type=str,
        choices=("mp16", "osv5m"),
        default="mp16",
        help="Which contrastive image adapter and cache root to use for the retrieval database.",
    )
    parser.add_argument(
        "--num-neighbors-image",
        type=int,
        default=32,
        help="Image-modality shortlist size for per_modality_union retrieval.",
    )
    parser.add_argument(
        "--num-neighbors-raw-image",
        type=int,
        default=16,
        help="Raw cached SigLIP-modality shortlist size for per_modality_union retrieval.",
    )
    parser.add_argument(
        "--num-neighbors-gps",
        type=int,
        default=16,
        help="GPS-modality shortlist size for per_modality_union retrieval.",
    )
    parser.add_argument(
        "--modality-pool-multiplier",
        type=int,
        default=1,
        help=(
            "For per_modality_union retrieval, fetch this many times more candidates per modality, "
            "then sample back down to the requested modality shortlist size during training."
        ),
    )
    parser.add_argument("--retrieval-chunk-size", type=int, default=65536)
    parser.add_argument("--reranker-hidden-dim", type=int, default=512)
    parser.add_argument("--reranker-num-heads", type=int, default=8)
    parser.add_argument("--reranker-num-layers", type=int, default=4)
    parser.add_argument("--reranker-ffn-dim", type=int, default=2048)
    parser.add_argument("--reranker-dropout", type=float, default=0.1)
    parser.add_argument(
        "--reranker-sequence-chunk-size",
        type=int,
        default=1024,
        help=(
            "How many query top-k lists to score per Transformer pass. "
            "Lower values reduce CUDA memory at the cost of some throughput."
        ),
    )
    parser.add_argument(
        "--disable-raw-image-features",
        action="store_true",
        help="Disable raw cached image query/candidate tokens in the reranker.",
    )
    parser.add_argument(
        "--disable-retrieval-features",
        action="store_true",
        help="Disable optional retrieval score/rank features in reranker input.",
    )
    parser.add_argument(
        "--use-osv-support-token",
        action="store_true",
        help="Add one fused nearby-OSV support token per candidate using cached street-view embeddings.",
    )
    parser.add_argument(
        "--osv-support-cache-root",
        type=str,
        default=None,
        help="Cache root for nearby OSV support lookups. Required when --use-osv-support-token is set.",
    )
    parser.add_argument(
        "--osv-support-neighbors-k",
        type=int,
        default=4,
        help="How many nearby OSV images to pool into each candidate support token.",
    )
    parser.add_argument(
        "--osv-support-radius-km",
        type=float,
        default=1.0,
        help="Maximum radius for nearby OSV support retrieval around each candidate.",
    )
    parser.add_argument(
        "--margin-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the margin ranking loss.",
    )
    parser.add_argument(
        "--margin-loss-margin",
        type=float,
        default=0.2,
        help="Margin for pairwise ranking: max(0, margin - (s_pos - s_neg)).",
    )
    parser.add_argument(
        "--inference-dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float32"),
        help="Autocast dtype for the frozen contrastive image encoder. Use float32 only when extra precision is needed.",
    )

    args = parser.parse_args()

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
    if args.retrieval_build_batch_size <= 0:
        raise ValueError("--retrieval-build-batch-size must be > 0")
    if args.num_neighbors_image < 0:
        raise ValueError("--num-neighbors-image must be >= 0")
    if args.num_neighbors_raw_image < 0:
        raise ValueError("--num-neighbors-raw-image must be >= 0")
    if args.num_neighbors_gps < 0:
        raise ValueError("--num-neighbors-gps must be >= 0")
    if args.num_neighbors_image <= 0 and args.num_neighbors_raw_image <= 0 and args.num_neighbors_gps <= 0:
        raise ValueError("At least one modality candidate budget must be > 0")
    if args.modality_pool_multiplier <= 0:
        raise ValueError("--modality-pool-multiplier must be > 0")
    if args.retrieval_chunk_size <= 0:
        raise ValueError("--retrieval-chunk-size must be > 0")
    if args.reranker_hidden_dim <= 0:
        raise ValueError("--reranker-hidden-dim must be > 0")
    if args.reranker_num_heads <= 0:
        raise ValueError("--reranker-num-heads must be > 0")
    if args.reranker_num_layers <= 0:
        raise ValueError("--reranker-num-layers must be > 0")
    if args.reranker_sequence_chunk_size <= 0:
        raise ValueError("--reranker-sequence-chunk-size must be > 0")
    if args.reranker_ffn_dim <= 0:
        raise ValueError("--reranker-ffn-dim must be > 0")
    if not (0.0 <= args.reranker_dropout < 1.0):
        raise ValueError("--reranker-dropout must be in [0, 1)")
    if args.osv_support_neighbors_k <= 0:
        raise ValueError("--osv-support-neighbors-k must be > 0")
    if args.osv_support_radius_km <= 0:
        raise ValueError("--osv-support-radius-km must be > 0")
    if args.margin_loss_weight < 0:
        raise ValueError("--margin-loss-weight must be >= 0")
    if args.margin_loss_margin <= 0:
        raise ValueError("--margin-loss-margin must be > 0")
    if args.margin_loss_weight == 0:
        raise ValueError("--margin-loss-weight must be > 0")

    contrastive_checkpoint_path = Path(args.contrastive_checkpoint_path).expanduser()
    if not contrastive_checkpoint_path.exists():
        raise FileNotFoundError(f"--contrastive-checkpoint-path not found: {contrastive_checkpoint_path}")
    args.contrastive_checkpoint_path = str(contrastive_checkpoint_path.resolve())
    if args.use_osv_support_token:
        if args.osv_support_cache_root is None:
            raise ValueError("--use-osv-support-token requires --osv-support-cache-root")
        osv_support_cache_root = Path(args.osv_support_cache_root).expanduser()
        if not osv_support_cache_root.exists():
            raise FileNotFoundError(f"--osv-support-cache-root not found: {osv_support_cache_root}")
        args.osv_support_cache_root = str(osv_support_cache_root.resolve())
    if not torch.cuda.is_available():
        raise RuntimeError("attention_reranker training requires CUDA.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("attention_reranker training requires a CUDA GPU with bfloat16 support.")

    L.seed_everything(args.seed, workers=True)
    np.random.seed(args.seed)

    resume_path, fit_ckpt_path = resolve_resume_training_state(args)
    run_dir, _, _ = prepare_experiment_dirs(args, resume_path)

    device_kwargs = trainer_device_kwargs(args.device)
    print(f"Trainer device config: {device_kwargs}")

    print("Loading embedding cache metadata...")
    (
        datamodule,
        embed_dim,
        train_datasets,
        val_datasets,
        cache_roots,
    ) = build_metadata_datamodule_and_dim(args, return_cache_context=True)
    print(f"Train datasets: {','.join(train_datasets)}")
    print(f"Val datasets: {','.join(val_datasets)}")
    if len(train_datasets) > 1:
        print(f"Train mixing: {args.train_mixing}")

    if args.retrieval_db_source not in cache_roots:
        available_sources = ", ".join(sorted(cache_roots))
        raise ValueError(
            f"--retrieval-db-source={args.retrieval_db_source} requires a matching cache root in the selected "
            f"datasets. Available sources: {available_sources}"
        )
    retrieval_cache_roots = {args.retrieval_db_source: cache_roots[args.retrieval_db_source]}

    retrieval_index_dir = ensure_attention_retrieval_index(
        contrastive_checkpoint_path=args.contrastive_checkpoint_path,
        cache_roots=retrieval_cache_roots,
        index_cache_dir=args.retrieval_index_dir,
        split=args.retrieval_split,
        val_pct=args.val_pct,
        build_batch_size=args.retrieval_build_batch_size,
        build_device=args.retrieval_build_device,
        build_dtype=args.retrieval_build_dtype,
        retrieval_db_source=args.retrieval_db_source,
    )
    print(f"Retrieval index dir: {retrieval_index_dir}")

    module = AttentionInterpolateLightningModule(
        embed_dim=embed_dim,
        contrastive_checkpoint_path=args.contrastive_checkpoint_path,
        retrieval_index_dir=str(retrieval_index_dir),
        retrieval_db_source=args.retrieval_db_source,
        lr=args.lr,
        weight_decay=args.weight_decay,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        num_neighbors_image=args.num_neighbors_image,
        num_neighbors_raw_image=args.num_neighbors_raw_image,
        num_neighbors_gps=args.num_neighbors_gps,
        modality_pool_multiplier=args.modality_pool_multiplier,
        retrieval_chunk_size=args.retrieval_chunk_size,
        reranker_hidden_dim=args.reranker_hidden_dim,
        reranker_num_heads=args.reranker_num_heads,
        reranker_num_layers=args.reranker_num_layers,
        reranker_ffn_dim=args.reranker_ffn_dim,
        reranker_dropout=args.reranker_dropout,
        reranker_sequence_chunk_size=args.reranker_sequence_chunk_size,
        use_raw_image_features=(not args.disable_raw_image_features),
        use_retrieval_features=(not args.disable_retrieval_features),
        use_osv_support_token=args.use_osv_support_token,
        osv_support_cache_root=args.osv_support_cache_root,
        osv_support_neighbors_k=args.osv_support_neighbors_k,
        osv_support_radius_km=args.osv_support_radius_km,
        margin_loss_weight=args.margin_loss_weight,
        margin_loss_margin=args.margin_loss_margin,
        inference_device=args.device,
        inference_dtype=args.inference_dtype,
    )
    print(f"Embedding dim: {module.embed_dim}")
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Geolocation reranker trainable parameters: {trainable_params:,}")

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
