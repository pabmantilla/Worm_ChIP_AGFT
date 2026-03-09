#!/usr/bin/env python3
"""Two-stage fine-tune of AlphaGenome on C. elegans ChIP-seq data.

Stage 1: Head-only training (frozen backbone) with AdamW via create_optimizer.
Stage 2: Full model fine-tuning (unfrozen backbone) with lower LR.

Usage:
  python scripts/finetune_chip.py --target gfp --config configs/chip_joint.json
  python scripts/finetune_chip.py --target polii --config configs/chip_joint.json
  python scripts/finetune_chip.py --target gfp  # resubmit: reloads saved args
"""

import os

import copy
import json
import pickle
import sys
import argparse
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import pearsonr, spearmanr

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fix alphagenome_ft import shadowing
_aft_real = PROJECT_ROOT / "alphagenome_ft" / "alphagenome_ft"
if str(_aft_real.parent) not in sys.path:
    sys.path.insert(0, str(_aft_real.parent))
if "alphagenome_ft" in sys.modules:
    _mod = sys.modules["alphagenome_ft"]
    if not hasattr(_mod, "CustomHead"):
        del sys.modules["alphagenome_ft"]

from alphagenome.models import dna_output
from alphagenome_research.model import dna_model

from alphagenome_ft import (
    HeadConfig,
    HeadType,
    register_custom_head,
    create_model_with_heads,
)
from alphagenome_ft.finetune.train import create_optimizer

from src import EncoderChIPHead, ChIPDataset, ChIPDataLoader

# ============================================================
# Constants
# ============================================================
HEAD_NAME = "chip_head"
RESULTS_BASE = PROJECT_ROOT / "results"
WINDOW_SIZE = 1000

DEFAULTS = {
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-6,
    "pooling_type": "mean",
    "nl_size": 1024,
    "dropout": 0.1,
    "activation": "relu",
    "early_stopping": 5,
    "reverse_complement": False,
    "reverse_complement_likelihood": 0.5,
    "aggregation": "log1p_sum",
    # Stage 2
    "stage2_lr": 1e-5,
    "stage2_epochs": 50,
    "stage2_patience": 5,
}


# ============================================================
# CLI & config
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune AlphaGenome on ChIP-seq")
    parser.add_argument("--target", required=True, choices=["gfp", "polii"])
    parser.add_argument("--name", default=None, help="Run name (default: chip_<target>)")
    parser.add_argument("--config", default=None)

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--nl-size", type=int, nargs='+', default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--activation", type=str, default=None, choices=["relu", "gelu"])
    parser.add_argument("--pooling-type", type=str, default=None, choices=["sum", "mean", "max"])

    parser.add_argument("--stage2-lr", type=float, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--stage2-patience", type=int, default=None)
    parser.add_argument("--skip-stage2", action="store_true")

    return parser.parse_args()


def load_config(config_path, defaults):
    if config_path is None:
        return dict(defaults)
    with open(config_path) as f:
        cfg = json.load(f)
    hp = dict(defaults)
    data = cfg.get("data", {})
    hp["batch_size"] = data.get("batch_size", hp["batch_size"])
    model = cfg.get("model", {})
    hp["pooling_type"] = model.get("pooling_type", hp["pooling_type"])
    nl = model.get("nl_size", hp["nl_size"])
    hp["nl_size"] = int(nl) if isinstance(nl, str) else nl
    hp["dropout"] = model.get("do", hp["dropout"])
    hp["activation"] = model.get("activation", hp["activation"])
    training = cfg.get("training", {})
    hp["num_epochs"] = training.get("num_epochs", hp["num_epochs"])
    hp["learning_rate"] = training.get("learning_rate", hp["learning_rate"])
    hp["weight_decay"] = training.get("weight_decay", hp["weight_decay"])
    hp["early_stopping"] = training.get("early_stopping_patience", hp["early_stopping"])
    two_stage = cfg.get("two_stage", {})
    hp["stage2_lr"] = two_stage.get("second_stage_lr", hp["stage2_lr"])
    hp["stage2_epochs"] = two_stage.get("second_stage_epochs", hp["stage2_epochs"])
    hp["stage2_patience"] = two_stage.get("early_stopping_patience", hp["stage2_patience"])
    # Data paths
    hp["fasta_path"] = str(PROJECT_ROOT / data.get("fasta_path", "genome/ce11.fa"))
    hp["bigwig_gfp"] = str(PROJECT_ROOT / data["bigwig_gfp"]) if "bigwig_gfp" in data else None
    hp["bigwig_polii"] = str(PROJECT_ROOT / data["bigwig_polii"]) if "bigwig_polii" in data else None
    hp["aggregation"] = data.get("aggregation", hp["aggregation"])
    return hp


def apply_cli_overrides(hp, args):
    if args.lr is not None: hp["learning_rate"] = args.lr
    if args.weight_decay is not None: hp["weight_decay"] = args.weight_decay
    if args.batch_size is not None: hp["batch_size"] = args.batch_size
    if args.epochs is not None: hp["num_epochs"] = args.epochs
    if args.patience is not None: hp["early_stopping"] = args.patience
    if args.nl_size is not None:
        hp["nl_size"] = args.nl_size if len(args.nl_size) > 1 else args.nl_size[0]
    if args.dropout is not None: hp["dropout"] = args.dropout
    if args.activation is not None: hp["activation"] = args.activation
    if args.pooling_type is not None: hp["pooling_type"] = args.pooling_type
    if args.stage2_lr is not None: hp["stage2_lr"] = args.stage2_lr
    if args.stage2_epochs is not None: hp["stage2_epochs"] = args.stage2_epochs
    if args.stage2_patience is not None: hp["stage2_patience"] = args.stage2_patience


# ============================================================
# Resume state persistence
# ============================================================

def save_training_state(checkpoint_dir, stage, epoch, best_valid_loss, best_epoch,
                        epochs_no_improve, train_loss_history, valid_loss_history,
                        opt_state, epoch1_preds=None, epoch1_targets=None,
                        best_preds=None, best_targets=None,
                        s1_completed=False, s1_metrics=None,
                        s1_train_loss_history=None, s1_valid_loss_history=None,
                        s1_best_epoch=None):
    state = {
        "stage": stage, "epoch": epoch,
        "best_valid_loss": best_valid_loss, "best_epoch": best_epoch,
        "epochs_no_improve": epochs_no_improve,
        "train_loss_history": [float(v) for v in train_loss_history],
        "valid_loss_history": [float(v) for v in valid_loss_history],
        "s1_completed": s1_completed, "s1_metrics": s1_metrics,
        "s1_train_loss_history": [float(v) for v in (s1_train_loss_history or [])],
        "s1_valid_loss_history": [float(v) for v in (s1_valid_loss_history or [])],
        "s1_best_epoch": s1_best_epoch,
    }
    with open(checkpoint_dir / "training_state.json", "w") as f:
        json.dump(state, f, indent=2)
    # Skip saving opt_state — it's ~1.5GB for 452M params and fills disk quota.
    # Resume will re-init optimizer and restart the current epoch.
    if epoch1_preds is not None:
        np.savez(checkpoint_dir / "epoch1_preds.npz", preds=epoch1_preds, targets=epoch1_targets)
    if best_preds is not None:
        np.savez(checkpoint_dir / "best_preds.npz", preds=best_preds, targets=best_targets)


def load_training_state(checkpoint_dir):
    state_file = checkpoint_dir / "training_state.json"
    if not state_file.exists():
        return None
    with open(state_file) as f:
        state = json.load(f)
    opt_file = checkpoint_dir / "opt_state.pkl"
    state["opt_state"] = pickle.load(open(opt_file, "rb")) if opt_file.exists() else None
    for tag in ["epoch1", "best"]:
        f = checkpoint_dir / f"{tag}_preds.npz"
        if f.exists():
            d = np.load(f)
            state[f"{tag}_preds"] = d["preds"]
            state[f"{tag}_targets"] = d["targets"]
        else:
            state[f"{tag}_preds"] = None
            state[f"{tag}_targets"] = None
    return state


# ============================================================
# Batch adapter + JIT train step factory
# ============================================================

def adapt_batch(batch, target_key):
    """Convert ChIPDataLoader batch to alphagenome_ft format."""
    return {
        "sequences": batch["seq"],
        "organism_index": batch["organism_index"],
        "negative_strand_mask": jnp.zeros(batch["seq"].shape[0], dtype=bool),
        f"targets_{HEAD_NAME}": batch[target_key],
    }


def make_train_step(model, optimizer, loss_fn, head_name, strand_reindexing):
    @jax.jit
    def train_step(params, state, opt_state, batch):
        def loss_fn_inner(p):
            predictions = model._predict(
                p, state,
                batch["sequences"], batch["organism_index"],
                negative_strand_mask=batch["negative_strand_mask"],
                strand_reindexing=strand_reindexing,
            )
            head_loss_dict = loss_fn(
                predictions[head_name],
                {"targets": batch[f"targets_{head_name}"],
                 "organism_index": batch["organism_index"]},
            )
            return head_loss_dict["loss"]
        loss_value, grads = jax.value_and_grad(loss_fn_inner)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_value
    return train_step


# ============================================================
# Metrics & figures
# ============================================================

def compute_metrics(preds, targets):
    r, _ = pearsonr(preds, targets)
    rho, _ = spearmanr(preds, targets)
    mse = float(np.mean((preds - targets) ** 2))
    return {"pearson_r": float(r), "spearman_rho": float(rho), "mse": mse}


def make_summary_figure(epoch1_preds, epoch1_targets,
                        best_preds, best_targets, metrics_best,
                        train_loss_hist, valid_loss_hist,
                        best_epoch, save_path, run_name):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    all_vals = [best_targets, best_preds]
    if epoch1_preds is not None:
        all_vals.append(epoch1_preds)
    vmin = min(v.min() for v in all_vals)
    vmax = max(v.max() for v in all_vals)
    lims = [vmin, vmax]

    if epoch1_preds is not None:
        m1 = compute_metrics(epoch1_preds, epoch1_targets)
        axes[0].scatter(epoch1_targets, epoch1_preds, alpha=0.1, s=1, rasterized=True)
        axes[0].plot(lims, lims, "r--", linewidth=0.5)
        axes[0].set_title(f"Epoch 1: r={m1['pearson_r']:.3f}, rho={m1['spearman_rho']:.3f}")
    else:
        axes[0].set_title("Epoch 1: N/A")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)

    axes[1].scatter(best_targets, best_preds, alpha=0.1, s=1, rasterized=True)
    axes[1].plot(lims, lims, "r--", linewidth=0.5)
    axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
    axes[1].set_title(f"Best (ep {best_epoch}): r={metrics_best['pearson_r']:.3f}, rho={metrics_best['spearman_rho']:.3f}")
    axes[1].set_xlim(lims); axes[1].set_ylim(lims)

    epochs_range = range(1, len(train_loss_hist) + 1)
    axes[2].plot(epochs_range, train_loss_hist, label="Train")
    axes[2].plot(epochs_range, valid_loss_hist, label="Valid")
    axes[2].axvline(best_epoch, color="gray", linestyle=":", alpha=0.7, label=f"Best (epoch {best_epoch})")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Loss (MSE)")
    axes[2].set_title("Training Loss"); axes[2].legend()

    plt.suptitle(run_name, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary figure saved to {save_path}")


def make_combined_summary(epoch1_preds, epoch1_targets,
                          final_preds, final_targets, final_metrics,
                          s1_train_loss, s1_valid_loss, s1_best_epoch,
                          s2_train_loss, s2_valid_loss, s2_best_epoch,
                          save_path, run_name):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    all_vals = [final_targets, final_preds]
    if epoch1_preds is not None:
        all_vals.append(epoch1_preds)
    vmin = min(v.min() for v in all_vals)
    vmax = max(v.max() for v in all_vals)
    lims = [vmin, vmax]

    if epoch1_preds is not None:
        m1 = compute_metrics(epoch1_preds, epoch1_targets)
        axes[0].scatter(epoch1_targets, epoch1_preds, alpha=0.1, s=1, rasterized=True)
        axes[0].plot(lims, lims, "r--", linewidth=0.5)
        axes[0].set_title(f"Epoch 1: r={m1['pearson_r']:.3f}, rho={m1['spearman_rho']:.3f}")
    else:
        axes[0].set_title("Epoch 1: N/A")
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)

    axes[1].scatter(final_targets, final_preds, alpha=0.1, s=1, rasterized=True)
    axes[1].plot(lims, lims, "r--", linewidth=0.5)
    axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
    best_label = f"S2 ep {s2_best_epoch}" if s2_train_loss else f"S1 ep {s1_best_epoch}"
    axes[1].set_title(f"Best ({best_label}): r={final_metrics['pearson_r']:.3f}, rho={final_metrics['spearman_rho']:.3f}")
    axes[1].set_xlim(lims); axes[1].set_ylim(lims)

    n_s1 = len(s1_train_loss)
    s1_epochs = list(range(1, n_s1 + 1))
    axes[2].plot(s1_epochs, s1_train_loss, color="tab:blue", label="S1 Train")
    axes[2].plot(s1_epochs, s1_valid_loss, color="tab:orange", label="S1 Valid")
    if s2_train_loss:
        s2_epochs = list(range(n_s1 + 1, n_s1 + len(s2_train_loss) + 1))
        axes[2].plot(s2_epochs, s2_train_loss, color="tab:blue", linestyle="--", label="S2 Train")
        axes[2].plot(s2_epochs, s2_valid_loss, color="tab:orange", linestyle="--", label="S2 Valid")
        axes[2].axvline(n_s1 + 0.5, color="red", linestyle="-", alpha=0.7, label="Unfreeze")
        axes[2].axvline(n_s1 + s2_best_epoch, color="green", linestyle=":", alpha=0.7, label=f"S2 best (ep {s2_best_epoch})")
    axes[2].axvline(s1_best_epoch, color="gray", linestyle=":", alpha=0.7, label=f"S1 best (ep {s1_best_epoch})")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Loss (MSE)")
    axes[2].set_title("Training Loss"); axes[2].legend(fontsize=8)

    plt.suptitle(run_name, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Combined summary saved to {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    target = args.target
    target_key = f"y_{target}"  # y_gfp or y_polii
    run_name = f"{args.name}_{target}" if args.name else f"chip_{target}"

    results_dir = RESULTS_BASE / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Persist args for resubmit
    args_file = results_dir / "args.json"
    has_cli_overrides = any([
        args.config, args.name, args.lr, args.weight_decay, args.batch_size,
        args.epochs, args.patience, args.nl_size, args.dropout,
        args.activation, args.pooling_type, args.stage2_lr,
        args.stage2_epochs, args.stage2_patience, args.skip_stage2,
    ])
    if has_cli_overrides or not args_file.exists():
        hp = load_config(args.config, DEFAULTS)
        apply_cli_overrides(hp, args)
        skip_stage2 = args.skip_stage2
        saved = {"hp": hp, "config": args.config, "target": target, "name": run_name, "skip_stage2": skip_stage2}
        with open(args_file, "w") as f:
            json.dump(saved, f, indent=2)
    else:
        with open(args_file) as f:
            saved = json.load(f)
        hp = saved["hp"]
        skip_stage2 = saved.get("skip_stage2", False)
        print(f"Loaded saved args from {args_file}")

    print(f"JAX devices: {jax.devices()}")
    print(f"Target: {target}")
    print(f"Results dir: {results_dir}")
    print(f"Hyperparameters: {json.dumps(hp, indent=2)}")

    # ---- Register head ----
    nl_size = hp["nl_size"] if isinstance(hp["nl_size"], list) else [hp["nl_size"]]
    register_custom_head(
        HEAD_NAME, EncoderChIPHead,
        HeadConfig(
            type=HeadType.GENOME_TRACKS,
            name=HEAD_NAME,
            output_type=dna_output.OutputType.RNA_SEQ,
            num_tracks=1,
            metadata={
                "pooling_type": hp["pooling_type"],
                "nl_size": nl_size,
                "do": hp["dropout"],
                "activation": hp["activation"],
            },
        ),
    )

    # ---- Model ----
    from huggingface_hub import snapshot_download
    hf_path = snapshot_download("google/alphagenome-all-folds")
    print(f"Weights: {hf_path}")

    model = create_model_with_heads(
        "all_folds", heads=[HEAD_NAME],
        checkpoint_path=hf_path,
        use_encoder_output=True,
        init_seq_len=WINDOW_SIZE,
    )
    model.freeze_except_head(HEAD_NAME)

    head_params = model.get_head_parameters(HEAD_NAME)
    head_count = sum(p.size for p in jax.tree_util.tree_leaves(head_params))
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Trainable (head only): {head_count:,}")

    # ---- Organism / strand setup ----
    organism_enum = dna_model.Organism.HOMO_SAPIENS
    strand_reindexing = jax.device_put(
        model._metadata[organism_enum].strand_reindexing,
        model._device_context._device,
    )

    # ---- Data ----
    fasta_path = hp.get("fasta_path", str(PROJECT_ROOT / "genome/ce11.fa"))

    # Always pass both bigwigs in correct positional order: gfp, polii
    gfp_path = hp.get("bigwig_gfp")
    polii_path = hp.get("bigwig_polii")
    if gfp_path is None or polii_path is None:
        raise ValueError("Both bigwig_gfp and bigwig_polii must be set in config")

    agg = hp["aggregation"]
    train_ds = ChIPDataset(fasta_path, gfp_path, polii_path, split="train",
                           window_size=WINDOW_SIZE,
                           reverse_complement=hp["reverse_complement"],
                           reverse_complement_likelihood=hp["reverse_complement_likelihood"],
                           aggregation=agg)
    val_ds = ChIPDataset(fasta_path, gfp_path, polii_path, split="val",
                         window_size=WINDOW_SIZE, aggregation=agg)
    test_ds = ChIPDataset(fasta_path, gfp_path, polii_path, split="test",
                          window_size=WINDOW_SIZE, aggregation=agg)

    train_loader = ChIPDataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True)
    val_loader = ChIPDataLoader(val_ds, batch_size=hp["batch_size"], shuffle=False)
    test_loader = ChIPDataLoader(test_ds, batch_size=hp["batch_size"], shuffle=False)

    print(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds)} samples, {len(val_loader)} batches")
    print(f"Test:  {len(test_ds)} samples, {len(test_loader)} batches")

    # ---- Setup step functions ----
    loss_fn = model.create_loss_fn_for_head(HEAD_NAME)
    optimizer = create_optimizer(
        model._params, trainable_head_names=[HEAD_NAME],
        learning_rate=hp["learning_rate"], weight_decay=hp["weight_decay"],
        heads_only=True,
    )
    opt_state = optimizer.init(model._params)
    train_step_fn = make_train_step(model, optimizer, loss_fn, HEAD_NAME, strand_reindexing)

    @jax.jit
    def eval_step(params, state, batch):
        predictions = model._predict(
            params, state,
            batch["sequences"], batch["organism_index"],
            negative_strand_mask=batch["negative_strand_mask"],
            strand_reindexing=strand_reindexing,
        )
        loss_dict = loss_fn(
            predictions[HEAD_NAME],
            {"targets": batch[f"targets_{HEAD_NAME}"],
             "organism_index": batch["organism_index"]},
        )
        return loss_dict["loss"]

    @jax.jit
    def predict_step(params, state, batch):
        predictions = model._predict(
            params, state,
            batch["sequences"], batch["organism_index"],
            negative_strand_mask=batch["negative_strand_mask"],
            strand_reindexing=strand_reindexing,
        )
        return predictions[HEAD_NAME]

    def do_train_step(raw_batch):
        nonlocal opt_state
        batch = adapt_batch(raw_batch, target_key)
        model._params, opt_state, loss_val = train_step_fn(
            model._params, model._state, opt_state, batch
        )
        return float(loss_val)

    def do_eval_step(raw_batch):
        batch = adapt_batch(raw_batch, target_key)
        return float(eval_step(model._params, model._state, batch))

    def do_predict_batch(raw_batch):
        batch = adapt_batch(raw_batch, target_key)
        preds = predict_step(model._params, model._state, batch)
        # preds shape: (batch, positions, num_tracks) — pool over positions
        # to get scalar prediction, matching how loss() pools
        if preds.ndim == 3:
            preds = jnp.mean(preds, axis=1)  # (batch, num_tracks)
        if preds.ndim == 2:
            preds = preds.squeeze(-1)  # (batch,)
        return np.array(preds, dtype=np.float32), np.array(raw_batch[target_key], dtype=np.float32)

    def collect_preds(loader):
        preds_list, targets_list = [], []
        for raw_batch in loader:
            p, t = do_predict_batch(raw_batch)
            preds_list.append(p)
            targets_list.append(t)
        return np.concatenate(preds_list), np.concatenate(targets_list)

    # ============================================================
    # Check for resume
    # ============================================================
    resume = load_training_state(checkpoint_dir)
    start_stage = 1
    start_epoch = 1

    if resume is not None:
        print(f"\nFound resume state: stage {resume['stage']}, epoch {resume['epoch']}")
        if resume["s1_completed"] and resume["stage"] == 2:
            start_stage = 2
        elif resume["s1_completed"]:
            start_stage = 3
        else:
            start_stage = 1
            start_epoch = resume["epoch"] + 1

    # ============================================================
    # Stage 1: Head-only
    # ============================================================
    if start_stage <= 1:
        print(f"\n{'='*60}")
        print(f"Stage 1: head-only (frozen encoder)")
        print(f"  LR={hp['learning_rate']}, WD={hp['weight_decay']}, BS={hp['batch_size']}")
        print(f"  Epochs={hp['num_epochs']}, Patience={hp['early_stopping']}")
        print(f"{'='*60}")

        if resume is not None and not resume["s1_completed"]:
            best_valid_loss = resume["best_valid_loss"]
            best_epoch = resume["best_epoch"]
            epochs_no_improve = resume["epochs_no_improve"]
            train_loss_history = resume["train_loss_history"]
            valid_loss_history = resume["valid_loss_history"]
            epoch1_preds = resume["epoch1_preds"]
            epoch1_targets = resume["epoch1_targets"]
            best_preds = resume["best_preds"]
            best_targets = resume["best_targets"]
            if resume["opt_state"] is not None:
                opt_state = resume["opt_state"]
                train_step_fn = make_train_step(model, optimizer, loss_fn, HEAD_NAME, strand_reindexing)
        else:
            best_valid_loss = float("inf")
            epochs_no_improve = 0
            best_epoch = 0
            best_preds = best_targets = epoch1_preds = epoch1_targets = None
            train_loss_history = []
            valid_loss_history = []

        best_params = None
        patience = hp["early_stopping"]

        with model._device_context:
            for epoch in range(start_epoch, hp["num_epochs"] + 1):
                train_losses = []
                for raw_batch in train_loader:
                    loss_val = do_train_step(raw_batch)
                    train_losses.append(loss_val)
                train_loss = np.mean(train_losses)

                valid_losses = []
                for raw_batch in val_loader:
                    valid_losses.append(do_eval_step(raw_batch))
                valid_loss = np.mean(valid_losses)

                train_loss_history.append(train_loss)
                valid_loss_history.append(valid_loss)

                print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f}", end="")

                if epoch == 1:
                    epoch1_preds, epoch1_targets = collect_preds(test_loader)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    epochs_no_improve = 0
                    best_epoch = epoch
                    best_preds, best_targets = collect_preds(test_loader)
                    best_params = copy.deepcopy(model._params)
                    model.save_checkpoint(str(checkpoint_dir / "best"), save_full_model=False)
                    print(" * (saved)")
                else:
                    epochs_no_improve += 1
                    print(f"  (no improve {epochs_no_improve}/{patience})")

                save_training_state(
                    checkpoint_dir, stage=1, epoch=epoch,
                    best_valid_loss=best_valid_loss, best_epoch=best_epoch,
                    epochs_no_improve=epochs_no_improve,
                    train_loss_history=train_loss_history,
                    valid_loss_history=valid_loss_history,
                    opt_state=opt_state,
                    epoch1_preds=epoch1_preds, epoch1_targets=epoch1_targets,
                    best_preds=best_preds, best_targets=best_targets,
                )

                if patience > 0 and epochs_no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

        if best_preds is None:
            with model._device_context:
                best_preds, best_targets = collect_preds(test_loader)
            best_epoch = len(train_loss_history)

        s1_metrics = compute_metrics(best_preds, best_targets)
        print(f"\nStage 1 Test (best epoch {best_epoch}): "
              f"r={s1_metrics['pearson_r']:.4f}, rho={s1_metrics['spearman_rho']:.4f}, MSE={s1_metrics['mse']:.4f}")

        make_summary_figure(
            epoch1_preds, epoch1_targets,
            best_preds, best_targets, s1_metrics,
            train_loss_history, valid_loss_history,
            best_epoch, results_dir / "summary_stage1.png",
            f"ChIP-{target.upper()} (stage 1)",
        )

        save_training_state(
            checkpoint_dir, stage=1, epoch=len(train_loss_history),
            best_valid_loss=best_valid_loss, best_epoch=best_epoch,
            epochs_no_improve=epochs_no_improve,
            train_loss_history=train_loss_history,
            valid_loss_history=valid_loss_history,
            opt_state=opt_state,
            epoch1_preds=epoch1_preds, epoch1_targets=epoch1_targets,
            best_preds=best_preds, best_targets=best_targets,
            s1_completed=True, s1_metrics=s1_metrics,
            s1_train_loss_history=train_loss_history,
            s1_valid_loss_history=valid_loss_history,
            s1_best_epoch=best_epoch,
        )
    else:
        s1_metrics = resume["s1_metrics"]
        train_loss_history = resume["s1_train_loss_history"]
        valid_loss_history = resume["s1_valid_loss_history"]
        best_epoch = resume["s1_best_epoch"]
        epoch1_preds = resume["epoch1_preds"]
        epoch1_targets = resume["epoch1_targets"]
        best_preds = resume["best_preds"]
        best_targets = resume["best_targets"]
        print(f"Stage 1 results (from previous run): r={s1_metrics['pearson_r']:.4f}")

    # ============================================================
    # Stage 2: Full model fine-tuning
    # ============================================================
    s2_metrics = None
    s2_best_preds = s2_best_targets = None
    s2_train_loss_history = []
    s2_valid_loss_history = []
    s2_best_epoch = 0

    if not skip_stage2 and start_stage <= 2:
        s2_start_epoch = 1
        s2_best_valid_loss = float("inf")
        s2_epochs_no_improve = 0
        s2_patience = hp["stage2_patience"]

        if resume is not None and resume["s1_completed"] and resume["stage"] == 2:
            s2_start_epoch = resume["epoch"] + 1
            s2_best_valid_loss = resume["best_valid_loss"]
            s2_best_epoch = resume["best_epoch"]
            s2_epochs_no_improve = resume["epochs_no_improve"]
            s2_train_loss_history = resume["train_loss_history"]
            s2_valid_loss_history = resume["valid_loss_history"]
            if resume["best_preds"] is not None:
                s2_best_preds = resume["best_preds"]
                s2_best_targets = resume["best_targets"]
        else:
            print(f"\n{'='*60}")
            print(f"Stage 2: Full model fine-tuning (unfrozen encoder)")
            print(f"  LR={hp['stage2_lr']}, Epochs={hp['stage2_epochs']}, Patience={s2_patience}")
            print(f"{'='*60}")

        if s2_start_epoch == 1:
            if 'best_params' in dir() and best_params is not None:
                model._params = best_params

        model.unfreeze_parameters(unfreeze_prefixes=['sequence_encoder'])
        print("Unfroze sequence_encoder parameters")

        s2_optimizer = create_optimizer(
            model._params, trainable_head_names=[HEAD_NAME],
            learning_rate=hp["stage2_lr"], weight_decay=hp["weight_decay"],
            heads_only=False,
        )
        if resume is not None and resume["s1_completed"] and resume["stage"] == 2 and resume["opt_state"] is not None:
            s2_opt_state = resume["opt_state"]
        else:
            s2_opt_state = s2_optimizer.init(model._params)

        loss_fn = model.create_loss_fn_for_head(HEAD_NAME)
        s2_train_step_fn = make_train_step(model, s2_optimizer, loss_fn, HEAD_NAME, strand_reindexing)

        @jax.jit
        def s2_eval_step(params, state, batch):
            predictions = model._predict(
                params, state,
                batch["sequences"], batch["organism_index"],
                negative_strand_mask=batch["negative_strand_mask"],
                strand_reindexing=strand_reindexing,
            )
            loss_dict = loss_fn(
                predictions[HEAD_NAME],
                {"targets": batch[f"targets_{HEAD_NAME}"],
                 "organism_index": batch["organism_index"]},
            )
            return loss_dict["loss"]

        with model._device_context:
            for epoch in range(s2_start_epoch, hp["stage2_epochs"] + 1):
                train_losses = []
                for raw_batch in train_loader:
                    batch = adapt_batch(raw_batch, target_key)
                    model._params, s2_opt_state, loss_val = s2_train_step_fn(
                        model._params, model._state, s2_opt_state, batch
                    )
                    train_losses.append(float(loss_val))
                s2_train_loss = np.mean(train_losses)

                valid_losses = []
                for raw_batch in val_loader:
                    batch = adapt_batch(raw_batch, target_key)
                    valid_losses.append(float(s2_eval_step(model._params, model._state, batch)))
                s2_valid_loss = np.mean(valid_losses)

                s2_train_loss_history.append(s2_train_loss)
                s2_valid_loss_history.append(s2_valid_loss)

                print(f"S2 Epoch {epoch:03d} | train_loss={s2_train_loss:.4f} | valid_loss={s2_valid_loss:.4f}", end="")

                if s2_valid_loss < s2_best_valid_loss:
                    s2_best_valid_loss = s2_valid_loss
                    s2_epochs_no_improve = 0
                    s2_best_epoch = epoch
                    s2_best_preds, s2_best_targets = collect_preds(test_loader)
                    model.save_checkpoint(str(checkpoint_dir / "best_stage2"), save_minimal_model=True)
                    print(" * (saved)")
                else:
                    s2_epochs_no_improve += 1
                    print(f"  (no improve {s2_epochs_no_improve}/{s2_patience})")

                save_training_state(
                    checkpoint_dir, stage=2, epoch=epoch,
                    best_valid_loss=s2_best_valid_loss, best_epoch=s2_best_epoch,
                    epochs_no_improve=s2_epochs_no_improve,
                    train_loss_history=s2_train_loss_history,
                    valid_loss_history=s2_valid_loss_history,
                    opt_state=s2_opt_state,
                    epoch1_preds=epoch1_preds, epoch1_targets=epoch1_targets,
                    best_preds=s2_best_preds, best_targets=s2_best_targets,
                    s1_completed=True, s1_metrics=s1_metrics,
                    s1_train_loss_history=train_loss_history,
                    s1_valid_loss_history=valid_loss_history,
                    s1_best_epoch=best_epoch,
                )

                if s2_patience > 0 and s2_epochs_no_improve >= s2_patience:
                    print(f"\nStage 2 early stopping at epoch {epoch}")
                    break

        if s2_best_preds is not None:
            s2_metrics = compute_metrics(s2_best_preds, s2_best_targets)
            print(f"\nStage 2 Test (best epoch {s2_best_epoch}): "
                  f"r={s2_metrics['pearson_r']:.4f}, rho={s2_metrics['spearman_rho']:.4f}, MSE={s2_metrics['mse']:.4f}")
            make_summary_figure(
                best_preds, best_targets,
                s2_best_preds, s2_best_targets, s2_metrics,
                s2_train_loss_history, s2_valid_loss_history,
                s2_best_epoch, results_dir / "summary_stage2.png",
                f"ChIP-{target.upper()} (stage 2)",
            )

    # ============================================================
    # Final summary
    # ============================================================
    final_metrics = s2_metrics if s2_metrics is not None else s1_metrics
    final_preds = s2_best_preds if s2_best_preds is not None else best_preds
    final_targets = s2_best_targets if s2_best_targets is not None else best_targets

    metrics_out = {
        "target": target,
        "config_path": args.config,
        "hyperparameters": hp,
        "stage1_test": s1_metrics,
        "stage1_best_epoch": best_epoch,
        "stage1_epochs_trained": len(train_loss_history),
        "stage2_test": s2_metrics,
        "stage2_best_epoch": s2_best_epoch,
        "stage2_epochs_trained": len(s2_train_loss_history),
        "best_epoch_test": final_metrics,
        "history": {
            "stage1_train_loss": [float(v) for v in train_loss_history],
            "stage1_valid_loss": [float(v) for v in valid_loss_history],
            "stage2_train_loss": [float(v) for v in s2_train_loss_history],
            "stage2_valid_loss": [float(v) for v in s2_valid_loss_history],
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Metrics saved to {results_dir / 'metrics.json'}")

    make_combined_summary(
        epoch1_preds, epoch1_targets,
        final_preds, final_targets, final_metrics,
        train_loss_history, valid_loss_history, best_epoch,
        s2_train_loss_history, s2_valid_loss_history, s2_best_epoch,
        results_dir / "summary_combined.png",
        f"ChIP-{target.upper()}",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
