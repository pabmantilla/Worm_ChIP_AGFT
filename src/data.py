"""
Data classes for AlphaGenome ChIP-seq finetuning on C. elegans.

Reads coverage from BigWig files over tiled 1000bp windows and returns
log1p(total_counts) as a scalar target per window.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pyBigWig
from pysam import FastaFile


# Chromosome split (~71/15/14 train/val/test)
CHROM_SPLITS = {
    "train": ["chrII", "chrV", "chrX"],
    "val": ["chrI"],
    "test": ["chrIII"],
}

SKIP_CHROMS = {"chrMtDNA", "chrIV", "chrM"}


class ChIPDataset:
    """Dataset that tiles 1000bp windows across C. elegans chromosomes
    and extracts log1p(sum coverage) from two BigWig files (GFP + PolII).

    Returns dict with keys:
        seq            - one-hot encoded DNA (window_size, 4)
        y_gfp          - scalar log1p(total GFP coverage)
        y_polii        - scalar log1p(total PolII coverage)
        organism_index - jnp.array([0])  (human idx; AG has no worm)
    """

    # One-hot mapping: A=0, C=1, G=2, T=3
    _BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3,
                     "a": 0, "c": 1, "g": 2, "t": 3}

    def __init__(
        self,
        fasta_path: str,
        bigwig_gfp_path: str,
        bigwig_polii_path: str,
        split: str = "train",
        window_size: int = 1000,
        reverse_complement: bool = False,
        reverse_complement_likelihood: float = 0.5,
        rng_key: jax.Array | None = None,
        aggregation: str = "log1p_sum",
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        assert aggregation in ("log1p_sum", "mean"), (
            f"aggregation must be 'log1p_sum' or 'mean', got {aggregation}"
        )
        self.fasta_path = fasta_path
        self.bigwig_gfp_path = bigwig_gfp_path
        self.bigwig_polii_path = bigwig_polii_path
        self.split = split
        self.window_size = window_size
        self.reverse_complement = reverse_complement
        self.reverse_complement_likelihood = reverse_complement_likelihood
        self._aggregation = aggregation

        if rng_key is None:
            self.rng_key = jax.random.PRNGKey(42)
        else:
            self.rng_key = rng_key

        # Open FASTA (pysam) — kept open for lifetime of dataset
        self._fasta = FastaFile(fasta_path)

        # Build window index: list of (chrom, start, end)
        self._windows = []
        chroms = CHROM_SPLITS[split]
        for chrom in chroms:
            chrom_len = self._fasta.get_reference_length(chrom)
            for start in range(0, chrom_len - window_size + 1, window_size):
                self._windows.append((chrom, start, start + window_size))

        print(f"ChIPDataset [{split}]: {len(self._windows)} windows "
              f"({window_size}bp) across {chroms}")

    def __len__(self):
        return len(self._windows)

    @staticmethod
    def _one_hot(seq: str) -> np.ndarray:
        """One-hot encode a DNA string to (L, 4) float32 array.
        Non-ACGT bases become all-zero rows.
        """
        mapping = ChIPDataset._BASE_TO_IDX
        arr = np.zeros((len(seq), 4), dtype=np.float32)
        for i, base in enumerate(seq):
            idx = mapping.get(base)
            if idx is not None:
                arr[i, idx] = 1.0
        return arr

    @staticmethod
    def _fetch_bigwig_agg(bw_path: str, chrom: str, start: int, end: int,
                          aggregation: str = "log1p_sum") -> float:
        """Aggregate BigWig values over a region.

        aggregation='log1p_sum': log1p(nansum) — for raw coverage.
        aggregation='mean':      nanmean       — for log2FC / ratio tracks.
        """
        bw = pyBigWig.open(bw_path)
        try:
            vals = bw.values(chrom, start, end)
            if aggregation == "mean":
                return float(np.nanmean(vals))
            else:
                return float(np.log1p(np.nansum(vals)))
        finally:
            bw.close()

    def _reverse_complement_onehot(self, seq_onehot: np.ndarray) -> np.ndarray:
        """Reverse complement: reverse sequence and swap A<->T, C<->G."""
        return seq_onehot[::-1, ::-1].copy()

    def __getitem__(self, idx):
        chrom, start, end = self._windows[idx]

        # Fetch sequence and one-hot encode
        seq_str = self._fasta.fetch(chrom, start, end)
        seq_ohe = self._one_hot(seq_str)

        # Optional reverse complement augmentation
        if self.reverse_complement:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            if jax.random.uniform(subkey) < self.reverse_complement_likelihood:
                seq_ohe = self._reverse_complement_onehot(seq_ohe)

        # Fetch BigWig targets
        y_gfp = self._fetch_bigwig_agg(self.bigwig_gfp_path, chrom, start, end, self._aggregation)
        y_polii = self._fetch_bigwig_agg(self.bigwig_polii_path, chrom, start, end, self._aggregation)

        return {
            "seq": jnp.array(seq_ohe),
            "y_gfp": y_gfp,
            "y_polii": y_polii,
            "organism_index": jnp.array([0]),
        }


class ChIPDataLoader:
    """Simple DataLoader that batches ChIPDataset samples into JAX arrays.

    Yields dicts with:
        seq            - (B, window_size, 4)
        y_gfp          - (B,)
        y_polii        - (B,)
        organism_index - (B,)
    """

    def __init__(
        self,
        dataset: ChIPDataset,
        batch_size: int = 32,
        shuffle: bool = False,
        rng_key: jax.Array | None = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng_key = rng_key
        if shuffle and rng_key is None:
            self.rng_key = jax.random.PRNGKey(42)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = jnp.arange(len(self.dataset))

        if self.shuffle:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            indices = jax.random.permutation(subkey, indices)

        num_batches = len(self)
        for i in range(num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(self.dataset))
            batch_indices = indices[start:end].tolist()

            samples = [self.dataset[int(j)] for j in batch_indices]
            yield self._stack_batch(samples)

    @staticmethod
    def _stack_batch(samples: list[dict]) -> dict:
        """Stack list of sample dicts into batched JAX arrays."""
        batch_seq = jnp.stack([s["seq"] for s in samples], axis=0)
        batch_y_gfp = jnp.array([s["y_gfp"] for s in samples])
        batch_y_polii = jnp.array([s["y_polii"] for s in samples])
        batch_org = jnp.stack([s["organism_index"] for s in samples], axis=0)
        batch_org = batch_org.squeeze(-1)

        return {
            "seq": batch_seq,
            "y_gfp": batch_y_gfp,
            "y_polii": batch_y_polii,
            "organism_index": batch_org,
        }
