from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger

try:
    from geographiclib.geodesic import Geodesic
except Exception:  # pragma: no cover - dependency fallback at runtime
    Geodesic = None

_WGS84_GEODESIC = Geodesic.WGS84 if Geodesic is not None else None
_DEFAULT_BACKBONE_MODEL_NAME = "google/siglip2-giant-opt-patch16-384"
_DEFAULT_BACKBONE_NAME = _DEFAULT_BACKBONE_MODEL_NAME.split("/")[-1]


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km. Inputs in degrees."""
    radius_km = 6371.0
    lat1, lon1, lat2, lon2 = (
        np.radians(lat1),
        np.radians(lon1),
        np.radians(lat2),
        np.radians(lon2),
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return radius_km * 2 * np.arcsin(np.sqrt(a))


def geodesic_km(lat1, lon1, lat2, lon2):
    """
    Vectorized WGS84 geodesic distance in km. Inputs in degrees.
    Falls back to haversine when geographiclib is unavailable.
    """
    if _WGS84_GEODESIC is None:
        return haversine_km(lat1, lon1, lat2, lon2)

    lat1_arr, lon1_arr, lat2_arr, lon2_arr = np.broadcast_arrays(
        np.asarray(lat1, dtype=np.float64),
        np.asarray(lon1, dtype=np.float64),
        np.asarray(lat2, dtype=np.float64),
        np.asarray(lon2, dtype=np.float64),
    )
    out = np.empty(lat1_arr.shape, dtype=np.float64)
    it = np.nditer(
        [lat1_arr, lon1_arr, lat2_arr, lon2_arr, out],
        flags=["refs_ok", "multi_index", "zerosize_ok"],
        op_flags=[["readonly"], ["readonly"], ["readonly"], ["readonly"], ["writeonly"]],
    )
    for src_lat, src_lon, dst_lat, dst_lon, out_cell in it:
        inv = _WGS84_GEODESIC.Inverse(
            float(src_lat),
            float(src_lon),
            float(dst_lat),
            float(dst_lon),
        )
        out_cell[...] = inv["s12"] / 1000.0
    return out


def wrap_longitude_deg_torch(lon_deg: torch.Tensor) -> torch.Tensor:
    """Wrap longitude to [-180, 180) for torch tensors."""
    return torch.remainder(lon_deg + 180.0, 360.0) - 180.0


def sanitize_latlon_deg_torch(latlon_deg: torch.Tensor) -> torch.Tensor:
    """Clamp/sanitize lat-lon tensors to valid ranges."""
    lat = torch.nan_to_num(latlon_deg[..., 0], nan=0.0, posinf=90.0, neginf=-90.0).clamp(-90.0, 90.0)
    lon = torch.nan_to_num(latlon_deg[..., 1], nan=0.0, posinf=180.0, neginf=-180.0)
    lon = wrap_longitude_deg_torch(lon)
    return torch.stack((lat, lon), dim=-1)


def model_name_tail(model_name: str) -> str:
    """Return the final path component of a model id like org/name."""
    tail = str(model_name).strip().rstrip("/").split("/")[-1].strip()
    if not tail or tail in {".", ".."}:
        raise ValueError(f"Invalid model name tail: {model_name!r}")
    return tail


def resolve_backbone_name(default_name: str = _DEFAULT_BACKBONE_NAME) -> str:
    """Resolve cache-path backbone name from BACKBONE_MODEL_NAME."""
    default_model_name = default_name if "/" in default_name else f"google/{default_name}"
    model_name = resolve_backbone_model_name(default_model_name=default_model_name)
    try:
        tail = model_name_tail(model_name)
    except ValueError:
        return default_name
    return tail


def resolve_backbone_model_name(default_model_name: str = _DEFAULT_BACKBONE_MODEL_NAME) -> str:
    """Resolve BACKBONE_MODEL_NAME into a model id."""
    raw_value = os.environ.get("BACKBONE_MODEL_NAME", "").strip()
    if not raw_value:
        return default_model_name
    model_name = raw_value.rstrip("/")
    if not model_name or model_name in {".", ".."}:
        return default_model_name
    if "/" in model_name:
        return model_name
    return f"google/{model_name}"


def infer_embed_dim(cache_root: str | Path, manifest_name: str = "manifest.json") -> int:
    manifest_path = Path(cache_root) / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    embed_dim = manifest.get("embedding_dim")
    if embed_dim is None:
        shapes = manifest.get("shapes", {})
        embed_shape = shapes.get("embeddings", [])
        if isinstance(embed_shape, list) and len(embed_shape) == 2:
            embed_dim = embed_shape[1]
    if embed_dim is None or int(embed_dim) <= 0:
        raise ValueError(f"Could not infer embedding_dim from {manifest_path}")
    return int(embed_dim)


def default_tb_run_name() -> str:
    """Return an anonymous timestamp-based TensorBoard run name."""
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    return timestamp


def _step_from_checkpoint_name(path: str | Path) -> int | None:
    """Parse step from files named ckpt_step{N}*.pt."""
    name = Path(path).name
    if not (name.startswith("ckpt_step") and name.endswith(".pt")):
        return None
    stem = name[len("ckpt_step") : -len(".pt")]
    digits = []
    for ch in stem:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        return None
    return int("".join(digits))


def prune_step_checkpoints(
    checkpoint_dir: str | Path,
    keep_last: int,
    protected_paths: Iterable[str | Path | None] | None = None,
) -> None:
    """Delete oldest ckpt_step*.pt files beyond retention limit."""
    if keep_last <= 0:
        return

    protected_resolved = set()
    if protected_paths:
        for path in protected_paths:
            if path is None:
                continue
            protected_resolved.add(Path(path).resolve())

    step_ckpts = []
    for path in Path(checkpoint_dir).glob("ckpt_step*.pt"):
        step = _step_from_checkpoint_name(path)
        if step is None:
            continue
        step_ckpts.append((step, path))

    if len(step_ckpts) <= keep_last:
        return

    step_ckpts.sort(key=lambda item: item[0])
    to_delete = []
    for _, path in step_ckpts[:-keep_last]:
        if path.resolve() in protected_resolved:
            continue
        to_delete.append(path)

    for path in to_delete:
        try:
            path.unlink()
        except FileNotFoundError:
            continue

    print(f"Pruned {len(to_delete)} old step checkpoints (keep_last={keep_last})")


def _read_logged_metric(trainer, keys, default=float("inf")):
    for key in keys:
        metric = trainer.callback_metrics.get(key)
        if metric is None:
            continue
        if torch.is_tensor(metric):
            return float(metric.detach().cpu().item())
        return float(metric)
    return default


class PeriodicCheckpointCallback(Callback):
    """Save step/latest/best/final checkpoints with legacy filenames."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        keep_last: int,
        pred_to_deg_fn: Callable[[Any, torch.Tensor], torch.Tensor | np.ndarray],
        val_metric_keys: Sequence[str] = ("val/median_km", "val_median_km"),
    ):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last = int(keep_last)
        self.pred_to_deg_fn = pred_to_deg_fn
        self.val_metric_keys = tuple(val_metric_keys)
        self.best_val_median_km = float("inf")
        self.best_ckpt_path = None

    def state_dict(self):
        return {"best_val_median_km": self.best_val_median_km}

    def load_state_dict(self, state_dict):
        self.best_val_median_km = float(state_dict.get("best_val_median_km", float("inf")))

    def on_fit_start(self, trainer, pl_module):
        module_best_buf = getattr(pl_module, "best_val_median_km", None)
        if module_best_buf is not None:
            module_best = float(module_best_buf.item())
            if np.isfinite(module_best):
                self.best_val_median_km = min(self.best_val_median_km, module_best)
        best_link = self.checkpoint_dir / "ckpt_best.pt"
        if best_link.is_symlink():
            self.best_ckpt_path = best_link.resolve()

    @staticmethod
    def _update_symlink(link_path, target_path):
        """Point link_path to target_path (relative symlink), replacing existing files."""
        link_path = Path(link_path)
        target_path = Path(target_path)
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        rel_target = os.path.relpath(target_path, start=link_path.parent)
        link_path.symlink_to(rel_target)

    def _compute_val_median_km(self, pl_module):
        preds = getattr(pl_module, "_val_preds", None)
        targets = getattr(pl_module, "_val_targets", None)
        if not preds or not targets:
            return float("inf")

        pred_tensor = torch.cat(preds, dim=0)
        pred_deg = self.pred_to_deg_fn(pl_module, pred_tensor)
        if torch.is_tensor(pred_deg):
            pred_deg = pred_deg.detach().cpu().numpy()
        else:
            pred_deg = np.asarray(pred_deg)

        tgt_np = torch.cat(targets, dim=0).detach().cpu().numpy()
        pred_lat = pred_deg[:, 0]
        pred_lon = pred_deg[:, 1]
        tgt_lat = tgt_np[:, 0]
        tgt_lon = tgt_np[:, 1]
        dists = geodesic_km(pred_lat, pred_lon, tgt_lat, tgt_lon)
        return float(np.median(dists))

    def _read_val_metric(self, trainer, pl_module):
        # Compute median-km from current epoch buffers to avoid step-lagged callback_metrics.
        current = self._compute_val_median_km(pl_module)
        if np.isfinite(current):
            return current

        metric = _read_logged_metric(
            trainer,
            keys=self.val_metric_keys,
            default=None,
        )
        if metric is not None:
            return float(metric)
        return float("inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        step = int(trainer.global_step)
        if step <= 0:
            return

        val_median_km = self._read_val_metric(trainer, pl_module)
        is_new_best = val_median_km < self.best_val_median_km
        if is_new_best:
            self.best_val_median_km = val_median_km

        val_loss = _read_logged_metric(trainer, keys=("val/loss", "val_loss"))
        val_metric_tag = "inf" if not np.isfinite(val_loss) else f"{val_loss:.4f}"
        ckpt_path = self.checkpoint_dir / f"ckpt_step{step:06d}_val_loss{val_metric_tag}.pt"
        latest_path = self.checkpoint_dir / "ckpt_latest.pt"
        trainer.save_checkpoint(str(ckpt_path))
        self._update_symlink(latest_path, ckpt_path)
        if is_new_best:
            best_path = self.checkpoint_dir / "ckpt_best.pt"
            self._update_symlink(best_path, ckpt_path)
            self.best_ckpt_path = ckpt_path.resolve()
            if trainer.is_global_zero:
                print(f"Saved new best checkpoint: {best_path} ({val_median_km:.1f} km)")

        prune_step_checkpoints(
            self.checkpoint_dir,
            self.keep_last,
            protected_paths=[self.best_ckpt_path],
        )
        if trainer.is_global_zero:
            print(f"Saved checkpoint: {ckpt_path}")

    def on_train_end(self, trainer, pl_module):
        final_path = self.checkpoint_dir / "ckpt_final.pt"
        trainer.save_checkpoint(str(final_path))
        if trainer.is_global_zero:
            print(f"Saved final model: {final_path}")


def trainer_device_kwargs(device_arg: str | None) -> dict[str, Any]:
    """Map --device to Lightning Trainer accelerator/devices kwargs."""
    if device_arg is None:
        return {"accelerator": "auto", "devices": 1}

    value = device_arg.lower()
    if value == "cpu":
        return {"accelerator": "cpu", "devices": 1}
    if value == "mps":
        return {"accelerator": "mps", "devices": 1}
    if value in {"cuda", "gpu"}:
        return {"accelerator": "gpu", "devices": 1}
    if value.startswith("cuda:"):
        idx = value.split(":", 1)[1]
        if not idx.isdigit():
            raise ValueError(f"Invalid --device value: {device_arg}")
        return {"accelerator": "gpu", "devices": [int(idx)]}
    raise ValueError(f"Unsupported --device value for Lightning: {device_arg}")


def _load_resume_checkpoint(path: str | Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    is_lightning = isinstance(ckpt, dict) and "pytorch-lightning_version" in ckpt and "state_dict" in ckpt
    if not is_lightning:
        raise ValueError(
            "--resume must point to a PyTorch Lightning checkpoint generated by this script."
        )
    return ckpt


def _override_head_args_from_resume(args: Any, resume_ckpt: dict[str, Any]) -> None:
    hparams = resume_ckpt.get("hyper_parameters", {})
    if not isinstance(hparams, dict):
        return

    for key in (
        "output_dim",
        "loc_hidden_dim",
        "loc_depth",
        "loc_expansion",
        "loc_dropout",
        "use_rff",
        "use_learned_rff",
        "rff_seed",
        "rff_features_list",
        "rff_sigmas_list",
        "logit_scale_init",
        "logit_scale_max",
        "logit_bias_init",
        "loss_weight_image_loc",
    ):
        if key not in hparams:
            continue
        saved_val = hparams[key]
        if not hasattr(args, key):
            continue
        current_val = getattr(args, key)
        if current_val != saved_val:
            print(f"Overriding --{key.replace('_', '-')} from {current_val} to {saved_val}")
            setattr(args, key, saved_val)


def resolve_resume_training_state(args: Any) -> tuple[Path | None, str | None]:
    """Apply resume overrides and return (resume_path, fit_ckpt_path)."""
    resume_path = None
    fit_ckpt_path = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        print(f"Resuming from checkpoint: {resume_path}")
        resume_ckpt = _load_resume_checkpoint(resume_path)
        _override_head_args_from_resume(args, resume_ckpt)
        resume_hparams = resume_ckpt.get("hyper_parameters", {})
        saved_warmup_steps = int(resume_hparams.get("warmup_steps", 0))
        if args.warmup_steps != saved_warmup_steps:
            print(f"Overriding --warmup-steps from {args.warmup_steps} to {saved_warmup_steps}")
            args.warmup_steps = saved_warmup_steps
        fit_ckpt_path = str(resume_path)
    return resume_path, fit_ckpt_path


def prepare_experiment_dirs(args: Any, resume_path: Path | None) -> tuple[Path, Path, Path]:
    """Resolve run/checkpoint/log directories, create them, and mirror path fields back to args."""
    if args.run_name:
        run_dir = Path(args.experiment_dir) / args.run_name
    elif resume_path is not None and resume_path.parent.name == "checkpoints":
        run_dir = resume_path.parent.parent
    else:
        run_dir = Path(args.experiment_dir) / default_tb_run_name()
    checkpoint_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"

    args.run_name = run_dir.name
    args.run_dir = str(run_dir)
    args.logs_dir = str(logs_dir)
    args.checkpoint_dir = str(checkpoint_dir)

    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Experiment run dir: {args.run_dir}")
    print(f"Logs dir: {args.logs_dir}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    return run_dir, checkpoint_dir, logs_dir


def create_default_tb_logger(run_dir: str | Path) -> TensorBoardLogger:
    run_dir = Path(run_dir)
    return TensorBoardLogger(
        save_dir=str(run_dir.parent),
        name=run_dir.name,
        default_hp_metric=False,
    )


def create_default_trainer(
    args: Any,
    *,
    logger: TensorBoardLogger,
    callbacks: Sequence[Callback],
    device_kwargs: dict[str, Any],
) -> L.Trainer:
    return L.Trainer(
        max_steps=args.steps,
        val_check_interval=args.val_every,
        check_val_every_n_epoch=None,
        limit_val_batches=args.val_batches,
        log_every_n_steps=args.log_every,
        logger=logger,
        callbacks=list(callbacks),
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        use_distributed_sampler=False,
        **device_kwargs,
    )


def log_run_config_to_tensorboard(trainer: L.Trainer, tb_logger: TensorBoardLogger, args: Any) -> None:
    if trainer.is_global_zero:
        tb_logger.experiment.add_text(
            "config",
            "\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())),
            global_step=0,
        )


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if device.type not in {"cpu", "cuda"}:
        raise RuntimeError(f"Unsupported device '{device_arg}'. Use cpu/cuda/auto.")
    return device


def _resolve_inference_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    value = dtype_arg.lower()
    if value == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32

    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if value not in mapping:
        allowed = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported inference dtype '{dtype_arg}'. Allowed: auto, {allowed}")
    dtype = mapping[value]
    if device.type == "cpu" and dtype == torch.float16:
        raise RuntimeError("float16 inference on CPU is not supported; use float32 or bfloat16")
    return dtype


def _to_feature_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0]
    if isinstance(output, dict):
        for key in ("image_embeds", "pooler_output", "last_hidden_state"):
            value = output.get(key)
            if value is not None:
                return value if key != "last_hidden_state" else value[:, 0]
    raise TypeError(f"Unsupported feature output type: {type(output)}")


def _resolve_default_checkpoint_path(module_file: str | Path) -> Path | None:
    module_path = Path(module_file).resolve()
    repo_root = module_path.parents[3]
    task_name = module_path.parent.name
    candidates = (
        repo_root / "exp" / task_name / "checkpoints" / "ckpt_best.pt",
        repo_root / "exp" / task_name / "checkpoints" / "ckpt_latest.pt",
        repo_root / "exp" / task_name / "checkpoints" / "ckpt_final.pt",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
