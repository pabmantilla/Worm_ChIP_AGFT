"""DeepSHAP-like attributions for AlphaGenome fine-tuned models (JAX).

Fixes three bugs in alphagenome_ft's compute_deepshap_attributions:
1. Shuffle: random adjacent swaps -> Altschul-Erickson dinucleotide-preserving
2. Base map: corrected to ACGT = 0,1,2,3
3. Attribution: grad difference -> integrated gradients per reference

Since JAX doesn't support backward hooks (needed for true DeepLIFT rescale
rule as in tangermeme/PyTorch), we use integrated gradients per reference:

    For each reference r_i, with M integration steps:
        attr_i = (x - r_i) * (1/M) Σ_{m=1}^{M} grad(f)(r_i + (m/M)*(x - r_i))

    Averaged over references. Satisfies the sum rule exactly as M -> inf.

Usage:
    from deepshap import deep_lift_shap
    attr = deep_lift_shap(model, seq, org, "mpra_head", n_shuffles=20)
"""

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# 1. Altschul-Erickson dinucleotide shuffle
# ---------------------------------------------------------------------------

def _onehot_to_indices(seq_onehot):
    """Convert one-hot (L, 4) to integer indices (L,). ACGT = 0,1,2,3."""
    return np.argmax(seq_onehot, axis=-1)


def _indices_to_onehot(indices, n_bases=4):
    """Convert integer indices (L,) to one-hot (L, 4)."""
    L = len(indices)
    oh = np.zeros((L, n_bases), dtype=np.float32)
    oh[np.arange(L), indices] = 1.0
    return oh


def dinucleotide_shuffle(seq_onehot, n=20, rng=None):
    """Dinucleotide-preserving shuffle via Altschul-Erickson (1988).

    Builds a directed graph of dinucleotide transitions and finds an Eulerian
    path, preserving exact dinucleotide frequencies.

    Args:
        seq_onehot: (seq_len, 4) numpy array, one-hot encoded.
        n: Number of shuffled references to generate.
        rng: numpy random Generator (or None for default).

    Returns:
        (n, seq_len, 4) numpy array of shuffled references.
    """
    if rng is None:
        rng = np.random.default_rng()

    seq_idx = _onehot_to_indices(seq_onehot)
    L = len(seq_idx)
    n_bases = 4
    results = np.empty((n, L, n_bases), dtype=np.float32)

    for i in range(n):
        results[i] = _indices_to_onehot(_shuffle_euler(seq_idx, n_bases, rng))

    return results


def _shuffle_euler(seq_idx, n_bases, rng):
    """Single Altschul-Erickson dinucleotide shuffle.

    Algorithm:
    1. Build adjacency lists for dinucleotide graph (each position is an edge).
    2. Shuffle the edges leaving each node (except last edge per node for connectivity).
    3. Follow Eulerian path from first base.
    """
    L = len(seq_idx)
    if L <= 2:
        return seq_idx.copy()

    # Build edge lists: for each base, list of (next_base, original_position)
    edges = [[] for _ in range(n_bases)]
    for i in range(L - 1):
        edges[seq_idx[i]].append(seq_idx[i + 1])

    # For Eulerian path: last edge from each node must connect to maintain
    # connectivity to the final base. We shuffle all but the last edge.
    # The last edge of each node's list is reserved.
    last_edges = [None] * n_bases
    for b in range(n_bases):
        if len(edges[b]) > 0:
            last_edges[b] = edges[b][-1]
            remaining = edges[b][:-1]
            rng.shuffle(remaining)
            edges[b] = list(remaining)
        else:
            edges[b] = []

    # Follow Eulerian path
    result = np.empty(L, dtype=seq_idx.dtype)
    result[0] = seq_idx[0]
    # Track position in each edge list
    edge_pos = [0] * n_bases

    for i in range(1, L):
        cur = result[i - 1]
        if edge_pos[cur] < len(edges[cur]):
            result[i] = edges[cur][edge_pos[cur]]
            edge_pos[cur] += 1
        else:
            # Use the reserved last edge
            result[i] = last_edges[cur]

    return result


# ---------------------------------------------------------------------------
# 2. Hypothetical attributions (tangermeme-style correction)
# ---------------------------------------------------------------------------

def hypothetical_attributions(multipliers, inp, refs):
    """Compute hypothetical attributions correcting for one-hot constraint.

    For each reference, for each position, the contribution of hypothetically
    placing base b there is: sum_c multiplier[pos, c] * (onehot(b, c) - ref[pos, c])
    averaged over references.

    Args:
        multipliers: (n_refs, seq_len, 4) gradient multipliers per reference.
        inp: (seq_len, 4) one-hot input.
        refs: (n_refs, seq_len, 4) reference sequences.

    Returns:
        (seq_len, 4) hypothetical attribution scores.
    """
    n_refs = refs.shape[0]
    seq_len = inp.shape[0]

    # For each hypothetical base b at each position:
    # hyp[pos, b] = mean_over_refs[ sum_c multiplier[ref, pos, c] * (I(c==b) - ref[pos, c]) ]
    hyp = np.zeros((seq_len, 4), dtype=np.float32)
    for r in range(n_refs):
        diff = np.eye(4, dtype=np.float32)[None, :, :] - refs[r][:, None, :]
        # diff shape: (seq_len, 4_hyp, 4_channel)
        # multipliers[r] shape: (seq_len, 4_channel)
        # For each hyp base b: sum over channels c of multiplier[pos,c] * diff[pos,b,c]
        contrib = np.einsum('sc,sbc->sb', multipliers[r], diff)
        hyp += contrib
    hyp /= n_refs
    return hyp


# ---------------------------------------------------------------------------
# 3. DeepLIFT/SHAP main function
# ---------------------------------------------------------------------------

def _build_compute_output(model, organism_index, head_name, output_index=None):
    """Build the scalar output function for JAX grad, reusing the pattern
    from custom_model.compute_input_gradients."""

    def compute_output(seq):
        if hasattr(model, '_custom_forward_fn') and model._custom_forward_fn is not None:
            rng_key = jax.random.PRNGKey(0)
            result = model._custom_forward_fn(
                model._params, model._state, rng_key, seq, organism_index,
            )
            if isinstance(result, tuple):
                predictions_dict, _ = result
            else:
                predictions_dict = result

            if isinstance(predictions_dict, dict):
                output = predictions_dict.get(head_name)
            else:
                output = None
        else:
            predictions = model._predict(
                model._params, model._state, seq, organism_index,
                negative_strand_mask=jnp.zeros((seq.shape[0],), dtype=jnp.bool_),
                strand_reindexing=jnp.array([], dtype=jnp.int32),
            )
            from alphagenome_ft.custom_model import _PredictionsDict
            if isinstance(predictions, _PredictionsDict):
                output = predictions._custom.get(head_name)
            elif hasattr(predictions, 'get'):
                output = predictions.get(head_name)
            else:
                output = None

        if output is None:
            raise ValueError(f"Head '{head_name}' not found in predictions.")

        if output_index is not None:
            output = output[..., output_index]
        elif output.ndim > 1:
            output = jnp.mean(output, axis=-1)

        return jnp.sum(output)

    return compute_output


def deep_lift_shap(
    model,
    sequence,
    organism_index,
    head_name,
    *,
    output_index=None,
    n_shuffles=20,
    n_steps=50,
    random_state=None,
    hypothetical=False,
):
    """DeepSHAP-like attributions via integrated gradients per reference.

    For each dinucleotide-shuffled reference r_i, computes integrated gradients:
        attr_i = (x - r_i) * (1/M) Σ_{m=1}^{M} grad(f)(r_i + (m/M)*(x - r_i))

    Averaged over all references. Satisfies the sum rule:
        sum(attr) = f(x) - mean(f(refs))   (exact as M -> inf)

    With hypothetical=True, applies the tangermeme-style correction so all 4
    bases get attribution at each position (useful for TF-MoDISco).

    Args:
        model: Loaded alphagenome_ft CustomModel.
        sequence: (1, seq_len, 4) jax array, one-hot.
        organism_index: (1,) jax array.
        head_name: Name of the custom head.
        output_index: Track index (None = mean of all tracks).
        n_shuffles: Number of dinucleotide-shuffled references.
        n_steps: Number of integration steps per reference (higher = more
            accurate sum rule, but more gradient evaluations).
        random_state: Int seed or numpy Generator.
        hypothetical: If True, return hypothetical attributions (4 channels
            active). If False, multiply by input one-hot (WT base only).

    Returns:
        (1, seq_len, 4) numpy float32 array of attributions.
    """
    if isinstance(random_state, int) or random_state is None:
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    # Get input as numpy for shuffle
    seq_np = np.array(sequence[0], dtype=np.float32)  # (seq_len, 4)
    seq_len = seq_np.shape[0]

    # Generate dinucleotide-shuffled references
    refs = dinucleotide_shuffle(seq_np, n=n_shuffles, rng=rng)  # (n, L, 4)

    # Build grad function
    compute_output = _build_compute_output(model, organism_index, head_name, output_index)
    grad_fn = jax.grad(compute_output)

    # Integrated gradients per reference:
    # For each ref, average gradients along the path ref -> input
    # alphas = m/M for m = 1, ..., M (exclude 0 = pure reference)
    alphas = np.linspace(1.0 / n_steps, 1.0, n_steps)  # (M,)

    multipliers_all = np.zeros((n_shuffles, seq_len, 4), dtype=np.float32)
    for i in range(n_shuffles):
        ref_jax = jnp.array(refs[i:i+1])  # (1, L, 4)
        grad_sum = np.zeros((seq_len, 4), dtype=np.float32)
        for alpha in alphas:
            interp = ref_jax + alpha * (sequence - ref_jax)  # (1, L, 4)
            grad_at_interp = grad_fn(interp)  # (1, L, 4)
            grad_sum += np.array(grad_at_interp[0], dtype=np.float32)
        multipliers_all[i] = grad_sum / n_steps

    if hypothetical:
        attr = hypothetical_attributions(multipliers_all, seq_np, refs)
    else:
        # attr_i = multiplier_i * (input - ref_i), averaged over references
        diffs = seq_np[None, :, :] - refs  # (n, L, 4)
        attr_per_ref = multipliers_all * diffs  # (n, L, 4)
        attr = np.mean(attr_per_ref, axis=0)  # (L, 4)

    return attr[None, :, :].astype(np.float32)  # (1, seq_len, 4)
