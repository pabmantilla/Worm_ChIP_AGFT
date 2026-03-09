"""
ChIP-seq prediction head for AlphaGenome fine-tuning.

Adapted from EncoderMPRAHead — uses raw encoder output (before transformer)
at 128bp resolution, pools to a scalar prediction per window.

Two independent instances are used: one for GFP, one for PolII.
"""

import sys
import importlib

import jax
import jax.numpy as jnp
import haiku as hk
from alphagenome_research.model import layers

# Work around namespace-package shadowing: when CWD contains the
# alphagenome_ft/ git clone directory (no __init__.py), Python sees
# it as a namespace package before the editable-install finder runs.
# Fix: temporarily ensure the real package path wins.
_aft_real = str(
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "alphagenome_ft" / "alphagenome_ft"
)
if _aft_real not in sys.path:
    sys.path.insert(0, str(__import__("pathlib").Path(_aft_real).parent))
# Force reimport if the wrong (namespace) version was cached
if "alphagenome_ft" in sys.modules:
    _mod = sys.modules["alphagenome_ft"]
    if not hasattr(_mod, "CustomHead"):
        del sys.modules["alphagenome_ft"]

from alphagenome_ft import CustomHead


class EncoderChIPHead(CustomHead):
    """Head that predicts a scalar log1p(counts) from encoder output.

    Architecture: LayerNorm -> hidden layers (with dropout + activation) -> Linear -> pool

    Configuration via metadata dict:
        pooling_type : str   — 'mean', 'sum', 'max' (default 'mean')
        nl_size      : int|list — hidden layer size(s) (default 1024)
        do           : float|None — dropout rate (default None)
        activation   : str   — 'relu' or 'gelu' (default 'relu')
    """

    ENCODER_RESOLUTION_BP = 128

    def __init__(self, *, name, output_type, num_tracks, num_organisms, metadata):
        super().__init__(
            name=name,
            num_tracks=num_tracks,
            output_type=output_type,
            num_organisms=num_organisms,
            metadata=metadata,
        )

        nl_size = metadata.get("nl_size", 1024) if metadata else 1024
        if isinstance(nl_size, int):
            self._hidden_sizes = [nl_size]
        elif isinstance(nl_size, list):
            self._hidden_sizes = nl_size
        else:
            raise ValueError(f"nl_size must be int or list, got {type(nl_size)}")

        self._do = metadata.get("do", None) if metadata else None

        pooling_type = metadata.get("pooling_type", "mean") if metadata else "mean"
        assert pooling_type in ("sum", "mean", "max"), (
            f"Invalid pooling type: {pooling_type}"
        )
        self._pooling_type = pooling_type

        activation = metadata.get("activation", "relu") if metadata else "relu"
        assert activation in ("relu", "gelu"), f"Invalid activation: {activation}"
        self._activation = activation

    def predict(self, embeddings, organism_index, **kwargs):
        """Predict from raw encoder output.

        Returns per-position predictions: (batch, seq_len//128, num_tracks).
        """
        if not hasattr(embeddings, "encoder_output"):
            raise AttributeError(
                "EncoderChIPHead requires 'encoder_output' in embeddings. "
                "Set use_encoder_output=True when creating the model."
            )
        if embeddings.encoder_output is None:
            raise ValueError("encoder_output is None.")

        x = embeddings.encoder_output  # (batch, positions, D)
        x = layers.LayerNorm(name="norm")(x)

        for i, hidden_size in enumerate(self._hidden_sizes):
            x = hk.Linear(hidden_size, name=f"hidden_{i}")(x)
            if self._do is not None:
                try:
                    rng_key = hk.next_rng_key()
                    x = hk.dropout(rng_key, self._do, x)
                except (RuntimeError, ValueError, AttributeError):
                    pass
            if self._activation == "gelu":
                x = jax.nn.gelu(x)
            else:
                x = jax.nn.relu(x)

        per_position_predictions = hk.Linear(self._num_tracks, name="output")(x)
        return per_position_predictions  # (batch, positions, num_tracks)

    def loss(self, predictions, batch):
        """MSE loss on scalar log1p(counts).

        Pools per-position predictions over all positions, then computes MSE
        against the target.

        Targets can be:
        - Per-position from BigWigDataModule: (batch, seq_len, num_tracks)
          → summed over positions and log1p-transformed to get a scalar.
        - Pre-computed scalar: (batch,) or (batch, 1)
          → used as-is.
        """
        targets = batch.get("targets")
        if targets is None:
            return {"loss": jnp.array(0.0)}

        # Pool predictions over sequence positions -> (batch, num_tracks)
        if self._pooling_type == "mean":
            pred_values = jnp.mean(predictions, axis=1)
        elif self._pooling_type == "max":
            pred_values = jnp.max(predictions, axis=1)
        elif self._pooling_type == "sum":
            pred_values = jnp.sum(predictions, axis=1)

        # Handle per-position targets from BigWigDataModule
        if targets.ndim == 3:
            # (batch, seq_len, num_tracks) -> sum coverage, log1p
            target_values = jnp.log1p(jnp.sum(jnp.nan_to_num(targets), axis=1))
        elif targets.ndim == 1:
            target_values = targets[:, None]
        else:
            target_values = targets

        mse_loss = jnp.mean((pred_values - target_values) ** 2)

        pred_flat = pred_values.flatten()
        targets_flat = target_values.flatten()
        pearson_corr = jnp.corrcoef(pred_flat, targets_flat)[0, 1]

        return {
            "loss": mse_loss,
            "mse": mse_loss,
            "pearson_corr": pearson_corr,
        }
