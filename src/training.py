"""
Training loop for ChIP-seq fine-tuning with metric tracking, plots, and summaries.

Wraps alphagenome_ft building blocks (optimizer, batch prep, JIT steps) and adds:
- Per-epoch metric capture (train/val/test loss + Pearson)
- Loss/Pearson plots saved after each stage
- training_summary.json with full history
- Two-stage support (heads-only → unfreeze encoder)
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from alphagenome.models import dna_model as ag_dna_model
from alphagenome_research.model import dna_model as research_dna_model

from alphagenome_ft.finetune.config import HeadSpec
from alphagenome_ft.finetune.data import BigWigDataModule, prepare_batch
from alphagenome_ft.finetune.train import create_optimizer
from alphagenome_ft.custom_model import CustomAlphaGenomeModel, load_checkpoint


def _make_plots(history: dict, output_dir: Path, stage_label: str) -> None:
    """Save loss and Pearson curves to output_dir."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    epochs = list(range(1, len(history["train_loss"]) + 1))

    # --- Loss plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="train")
    if history["val_loss"]:
        ax.plot(epochs, history["val_loss"], label="val")
    if history["test_loss"]:
        ax.plot(epochs, history["test_loss"], label="test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(f"{stage_label} — Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"loss_{stage_label.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)

    # --- Pearson plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    if history["train_pearson"]:
        ax.plot(epochs[: len(history["train_pearson"])], history["train_pearson"], label="train")
    if history["val_pearson"]:
        ax.plot(epochs[: len(history["val_pearson"])], history["val_pearson"], label="val")
    if history["test_pearson"]:
        ax.plot(epochs[: len(history["test_pearson"])], history["test_pearson"], label="test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"{stage_label} — Pearson Correlation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"pearson_{stage_label.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {output_dir}")


def _save_summary(
    output_dir: Path,
    target: str,
    hyperparams: dict,
    stage_histories: dict[str, dict],
    best_epoch: int,
    best_val_loss: float,
) -> None:
    """Write training_summary.json."""
    # Merge histories from all stages
    combined = {
        "train_loss": [],
        "train_pearson": [],
        "val_loss": [],
        "val_pearson": [],
        "test_loss": [],
        "test_pearson": [],
    }
    for _stage, hist in stage_histories.items():
        for key in combined:
            combined[key].extend(hist.get(key, []))

    summary = {
        "target": target,
        "hyperparameters": hyperparams,
        "stages": {name: hist for name, hist in stage_histories.items()},
        "history": combined,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
    path = output_dir / "training_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {path}")


def _evaluate_split(
    model: CustomAlphaGenomeModel,
    data_module: BigWigDataModule,
    split: str,
    head_names: list[str],
    eval_step_fn,
    organism_index_value: int,
) -> dict[str, float]:
    """Run eval on a split, return {loss, pearson} per head."""
    losses = {h: [] for h in head_names}
    all_preds = {h: [] for h in head_names}
    all_targets = {h: [] for h in head_names}

    for batch_np in data_module.iter_batches(split):
        batch = prepare_batch(batch_np, organism_index_value, head_names)
        head_losses, head_preds = eval_step_fn(model._params, model._state, batch)
        for h in head_names:
            losses[h].append(float(head_losses[h]))
            all_preds[h].append(np.array(head_preds[h]))
            all_targets[h].append(np.array(batch[f"targets_{h}"]))

    if not losses[head_names[0]]:
        return {}

    # Compute dataset-level Pearson (not batch-averaged)
    metrics: dict[str, float] = {}
    total_loss = 0.0
    for h in head_names:
        avg_loss = float(np.mean(losses[h]))
        total_loss += avg_loss
        metrics[f"{h}_loss"] = avg_loss

        preds_cat = np.concatenate(all_preds[h], axis=0)  # (N, pos, tracks)
        tgts_cat = np.concatenate(all_targets[h], axis=0)  # (N, seq_len, tracks)

        # Pool preds (mean over positions) and targets (sum + log1p)
        pred_scalar = preds_cat.mean(axis=1).flatten()
        tgt_scalar = np.log1p(np.nansum(tgts_cat, axis=1)).flatten()

        if len(pred_scalar) > 1:
            pearson = float(np.corrcoef(pred_scalar, tgt_scalar)[0, 1])
        else:
            pearson = 0.0
        metrics[f"{h}_pearson"] = pearson

    metrics["loss"] = total_loss
    return metrics


def _run_stage(
    model: CustomAlphaGenomeModel,
    data_module: BigWigDataModule,
    head_specs: Sequence[HeadSpec],
    *,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    heads_only: bool,
    seed: int,
    early_stopping_patience: int,
    organism: str,
    output_dir: Path,
    stage_label: str,
    verbose: bool = False,
) -> tuple[dict, int, float]:
    """Run one training stage with metric capture.

    Returns (history_dict, best_epoch, best_val_loss).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"

    head_names = [s.head_id for s in head_specs]

    # Optimizer
    optimizer = create_optimizer(
        model._params, head_names, learning_rate, weight_decay, heads_only
    )
    opt_state = optimizer.init(model._params)

    # Organism setup
    organism_enum = getattr(ag_dna_model.Organism, organism)
    organism_index_value = research_dna_model.convert_to_organism_index(organism_enum)
    strand_reindexing = jax.device_put(
        model._metadata[organism_enum].strand_reindexing,
        model._device_context._device,
    )

    loss_fns = {name: model.create_loss_fn_for_head(name) for name in head_names}

    # JIT-compiled steps
    @jax.jit
    def train_step(params, state, current_opt_state, batch):
        def loss_fn(p):
            preds = model._predict(
                p, state, batch["sequences"], batch["organism_index"],
                negative_strand_mask=batch["negative_strand_mask"],
                strand_reindexing=strand_reindexing,
            )
            total = 0.0
            for h in head_names:
                ld = loss_fns[h](
                    preds[h],
                    {"targets": batch[f"targets_{h}"], "organism_index": batch["organism_index"]},
                )
                total = total + ld["loss"]
            return total
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = optimizer.update(grads, current_opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss_val

    @jax.jit
    def eval_step(params, state, batch):
        preds = model._predict(
            params, state, batch["sequences"], batch["organism_index"],
            negative_strand_mask=batch["negative_strand_mask"],
            strand_reindexing=strand_reindexing,
        )
        head_losses = {}
        head_preds = {}
        for h in head_names:
            ld = loss_fns[h](
                preds[h],
                {"targets": batch[f"targets_{h}"], "organism_index": batch["organism_index"]},
            )
            head_losses[h] = ld["loss"]
            head_preds[h] = preds[h]
        return head_losses, head_preds

    # Count steps
    train_intervals = list(data_module._intervals.get("train", ()))
    n_train = len(train_intervals)
    if data_module._drop_last:
        steps_per_epoch = n_train // data_module._batch_size
    else:
        steps_per_epoch = math.ceil(n_train / data_module._batch_size)

    has_valid = "valid" in data_module._intervals and len(data_module._intervals["valid"]) > 0
    has_test = "test" in data_module._intervals and len(data_module._intervals["test"]) > 0

    # History
    history: dict[str, list] = {
        "train_loss": [], "train_pearson": [],
        "val_loss": [], "val_pearson": [],
        "test_loss": [], "test_pearson": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    print(f"\n{'=' * 60}")
    print(f"{stage_label}")
    print(f"{'=' * 60}")
    print(f"  LR={learning_rate}, WD={weight_decay}, epochs={num_epochs}")
    print(f"  {n_train} train examples, {steps_per_epoch} steps/epoch")
    print(f"  heads_only={heads_only}, patience={early_stopping_patience}")
    print(f"{'=' * 60}")

    with model._device_context:
        for epoch in range(1, num_epochs + 1):
            t0 = time.time()

            # --- Train ---
            train_losses = []
            for step, batch_np in enumerate(
                data_module.iter_batches("train", seed=seed + epoch), 1
            ):
                batch = prepare_batch(batch_np, organism_index_value, head_names)
                model._params, opt_state, loss_val = train_step(
                    model._params, model._state, opt_state, batch
                )
                train_losses.append(float(loss_val))
                if verbose and step % 50 == 0:
                    print(f"    step {step}/{steps_per_epoch} loss={float(loss_val):.4f}", end="\r", flush=True)

            train_loss_avg = float(np.mean(train_losses)) if train_losses else 0.0
            history["train_loss"].append(train_loss_avg)

            # --- Evaluate train Pearson (dataset-level) ---
            train_metrics = _evaluate_split(
                model, data_module, "train", head_names, eval_step, organism_index_value
            )
            train_pearson = train_metrics.get(f"{head_names[0]}_pearson", 0.0)
            history["train_pearson"].append(train_pearson)

            # --- Validation ---
            val_loss = None
            val_pearson = None
            if has_valid:
                val_metrics = _evaluate_split(
                    model, data_module, "valid", head_names, eval_step, organism_index_value
                )
                val_loss = val_metrics.get("loss", 0.0)
                val_pearson = val_metrics.get(f"{head_names[0]}_pearson", 0.0)
                history["val_loss"].append(val_loss)
                history["val_pearson"].append(val_pearson)

            # --- Test ---
            test_loss = None
            test_pearson = None
            if has_test:
                test_metrics = _evaluate_split(
                    model, data_module, "test", head_names, eval_step, organism_index_value
                )
                test_loss = test_metrics.get("loss", 0.0)
                test_pearson = test_metrics.get(f"{head_names[0]}_pearson", 0.0)
                history["test_loss"].append(test_loss)
                history["test_pearson"].append(test_pearson)

            elapsed = time.time() - t0

            # --- Print ---
            line = f"  Epoch {epoch}/{num_epochs} ({elapsed:.0f}s) | train_loss={train_loss_avg:.4f} train_r={train_pearson:.4f}"
            if val_loss is not None:
                line += f" | val_loss={val_loss:.4f} val_r={val_pearson:.4f}"
            if test_loss is not None:
                line += f" | test_loss={test_loss:.4f} test_r={test_pearson:.4f}"
            print(line)

            # --- Checkpoint + early stopping ---
            track_loss = val_loss if val_loss is not None else train_loss_avg
            if track_loss < best_val_loss:
                best_val_loss = track_loss
                best_epoch = epoch
                epochs_no_improve = 0
                print(f"    -> New best (loss={best_val_loss:.6f}), saving checkpoint")
                model.save_checkpoint(str(checkpoint_dir), save_full_model=False)
            else:
                epochs_no_improve += 1

            if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
                print(f"  Early stopping after {epochs_no_improve} epochs without improvement")
                break

    # Save plots for this stage
    _make_plots(history, output_dir, stage_label)

    print(f"\n  {stage_label} complete — best epoch {best_epoch}, best loss {best_val_loss:.6f}")
    return history, best_epoch, best_val_loss


def run_training(
    model: CustomAlphaGenomeModel,
    data_module: BigWigDataModule,
    head_specs: Sequence[HeadSpec],
    *,
    target: str,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    seed: int = 42,
    early_stopping_patience: int = 5,
    organism: str = "HOMO_SAPIENS",
    results_dir: Path,
    second_stage_lr: float | None = None,
    second_stage_epochs: int = 50,
    verbose: bool = False,
    hyperparams: dict | None = None,
) -> dict:
    """Full training pipeline with optional two-stage training.

    Stage 1: heads_only (frozen backbone).
    Stage 2 (optional): unfreeze encoder, lower LR.

    Saves per-stage plots, checkpoints, and a final training_summary.json.

    Returns combined history dict.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    stage_histories: dict[str, dict] = {}
    overall_best_epoch = 0
    overall_best_val_loss = float("inf")

    head_name = head_specs[0].head_id

    # ---- Stage 1: heads-only ----
    print("\nFreezing backbone for Stage 1...")
    model.freeze_except_head(head_name)

    s1_dir = results_dir / "stage1"
    s1_hist, s1_best_epoch, s1_best_loss = _run_stage(
        model, data_module, head_specs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        heads_only=True,
        seed=seed,
        early_stopping_patience=early_stopping_patience,
        organism=organism,
        output_dir=s1_dir,
        stage_label="Stage 1 (heads only)",
        verbose=verbose,
    )
    stage_histories["stage1"] = s1_hist
    overall_best_epoch = s1_best_epoch
    overall_best_val_loss = s1_best_loss

    # ---- Stage 2: unfreeze encoder ----
    if second_stage_lr is not None:
        print("\n\nLoading best Stage 1 checkpoint and unfreezing encoder...")
        best_ckpt = s1_dir / "checkpoint"
        if best_ckpt.exists():
            loaded = load_checkpoint(
                str(best_ckpt),
                base_model_version="all_folds",
                init_seq_len=data_module._intervals["train"][0].end - data_module._intervals["train"][0].start,
            )
            model._params = loaded._params
            model._state = loaded._state
            print("  Checkpoint loaded")

        # Unfreeze encoder parameters for full fine-tuning
        model.unfreeze_parameters(unfreeze_prefixes=["sequence_encoder"])

        s2_dir = results_dir / "stage2"
        s2_hist, s2_best_epoch, s2_best_loss = _run_stage(
            model, data_module, head_specs,
            learning_rate=second_stage_lr,
            weight_decay=weight_decay,
            num_epochs=second_stage_epochs,
            heads_only=False,
            seed=seed + 1000,
            early_stopping_patience=early_stopping_patience,
            organism=organism,
            output_dir=s2_dir,
            stage_label="Stage 2 (encoder unfrozen)",
            verbose=verbose,
        )
        stage_histories["stage2"] = s2_hist
        if s2_best_loss < overall_best_val_loss:
            overall_best_val_loss = s2_best_loss
            overall_best_epoch = num_epochs + s2_best_epoch

    # ---- Final summary + combined plots ----
    hp = hyperparams or {
        "target": target,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "second_stage_lr": second_stage_lr,
        "second_stage_epochs": second_stage_epochs,
        "early_stopping_patience": early_stopping_patience,
    }
    _save_summary(results_dir, target, hp, stage_histories, overall_best_epoch, overall_best_val_loss)

    # Combined plot across all stages
    combined: dict[str, list] = {
        "train_loss": [], "train_pearson": [],
        "val_loss": [], "val_pearson": [],
        "test_loss": [], "test_pearson": [],
    }
    for hist in stage_histories.values():
        for k in combined:
            combined[k].extend(hist.get(k, []))
    _make_plots(combined, results_dir, "Combined")

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"  Best epoch: {overall_best_epoch}")
    print(f"  Best val loss: {overall_best_val_loss:.6f}")
    print(f"  Results: {results_dir}")
    print(f"{'=' * 60}")

    return combined
