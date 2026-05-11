# fractalsort_cpu

A CPU adaptation of the [FractalSort algorithm](https://ieeexplore.ieee.org/abstract/document/11348110/), originally designed for FPGA/hardware accelerators. This project brings FractalSort to the CPU for accessibility and broader experimentation. It adopts a histogram merge tree index for sorting and querying/retrieval, achieving lower DRAM bandwidth than radix sort by decomposing keys into MSB-based bins with compact entries and per-batch sorted runs.

## Architecture

FractalSort decomposes each p-bit key into two parts:

```
key (p bits):
  ├─ top (ln-lb) bits      → bin_id  (MSB, determines which bin)
  └─ bottom entry_bits      → entry   (lb + (p-ln) bits, stored per key)
```

Where `ln = ceil(log2(n))`, `lb` controls bin size, and `entry_bits = lb + (p - ln)`.

For small precisions (`p <= 20`), a direct histogram mode is used instead — no bins or scatter, just a counting histogram with O(n + 2^p) reconstruction.

### Phases

1. **Process**: Single-pass direct scatter. For each key, extract `bin_id` from MSBs and `entry` from the remaining bits. Write entry to the bin's region in `sbatch_mem`. Keys are processed in batches, with each batch producing a sorted run per bin.

2. **Sort**: Radix-sort (or insertion sort for small bins) entries within each bin per batch. Sorted runs are concatenated — no global index array is needed.

3. **Get item**: Segment tree walk over bin counts to find the bin (O(log n_bins)). K-way selection across sorted runs via binary search to find the entry at the target rank. Reconstruct key as `(bin_id << entry_bits) | entry`.

4. **Reconstruct all**: K-way merge of sorted runs across all bins to produce the full sorted output.

### Optimal lb Selection

The `lb` parameter controls the trade-off between bin count and entry size:

| Rule | Bins | Use case |
|------|------|----------|
| `lb = e - 10` | 1024 | Default for e <= 20 |
| `lb = e - 6` | 64 | Default for e > 20, fewer bins |

## Requirements

- Python 3.8+
- NumPy
- Numba

```
pip install numpy numba
```

## Usage

### Sort and access

```python
from fractalsort_cpu import fractalsort

import numpy as np

# Generate random 32-bit keys
keys = np.random.randint(0, 2**32, size=1_000_000, dtype=np.uint32)

# Sort (first call includes JIT compilation)
result = fractalsort(keys, p=32, lb=12)

# Access sorted keys by position
smallest = result[0]
largest = result[-1]
median = result[len(result) // 2]

# Reconstruct all sorted keys
sorted_keys = result.reconstruct_all()
assert np.array_equal(sorted_keys, np.sort(keys))
```

### Parameters

```python
result = fractalsort(
    keys,           # uint32 array of keys
    p=32,           # key precision in bits
    lb=None,        # log2(bin size), default: e-10 (e<=20) or e-6 (e>20)
    n_batches=4,    # processing batches (for streaming)
)
```

### Result object

```python
result.get_item(position)    # O(log bins + k*log(bin_size)) point query
result[i]                    # same via __getitem__
result[10:20]                # slice access
len(result)                  # total number of keys
result.reconstruct_all()     # all keys in sorted order
```

### Internal arrays (for advanced use)

```python
result.sbatch_mem         # entry array (per-bin regions, sorted runs)
result.bin_counts         # entries per bin
result.bin_cumulative     # cumulative bin starts
result.batch_boundaries   # [n_bins, n_batches+1] run boundaries per bin
result.n_batches          # number of batches
result.ln                 # tree depth
result.lb                 # log2(bin size)
result.entry_bits         # bits per entry
result.n_bins             # number of bins
result.seg_tree           # segment tree for O(log n_bins) bin lookup
```

## Testing

```
python test_fractalsort.py [e] [lb]
```

Examples:
```
python test_fractalsort.py          # e=18, auto lb
python test_fractalsort.py 20       # e=20, auto lb
python test_fractalsort.py 20 12    # e=20, lb=12
```

## Performance

Benchmarked on a single core (numba JIT), p=32:

| e | n | lb | bins | frmw M/s | radix M/s | ratio |
|---|---|-----|------|----------|-----------|-------|
| 18 | 262K | 10 | 256 | 124 | 57 | 0.46x |
| 22 | 4.2M | 14 | 256 | 76 | 59 | 0.77x |
| 24 | 16.8M | 16 | 256 | 98 | 67 | 0.69x |
| 26 | 67.1M | 20 | 64 | 122 | 71 | 0.59x |
| 30 | 1.07B | 22 | 256 | 78 | 43 | 0.55x |

## License

See LICENSE file.
