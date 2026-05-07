# fractalsort_cpu

A CPU adaptation of the [FractalSort](https://ieeexplore.ieee.org/abstract/document/11348110/) algorithm, originally designed for FPGA/hardware accelerators. This project brings FractalSort to the CPU for accessibility and broader experimentation. It adopts the same histogram merge tree index for sorting and querying/retrieval, achieving lower DRAM bandwidth than radix sort by decomposing keys into a tree-ordered bin structure with compact trailing-bit entries.

## Architecture

FractalSort decomposes each p-bit key into three parts:

```
key (p bits):
  ├─ bits 0..ln-lb-1      → bin_id  (tree-ordered, determines which bin)
  ├─ bits ln-lb..ln-1      → offset  (lb bits, encoded by sorted position within bin)
  └─ bits ln..p-1          → trailing (p-ln bits, stored in entry)
```

Where `ln = ceil(log2(n))` and `lb = log2(b)` controls bin size.

### Phases

1. **Process**: Single-pass direct scatter. For each key, compute `bin_id` (a register scalar, not stored), compute `entry = tree_offset | trailing`, write entry to the bin's region in `sbatch_mem`. Histogram updated in cache. DRAM: 4B read + entry write.

2. **Finalize**: Radix-sort entries within each bin by entry value. Produces `index_array` mapping `sorted_position → arrival_position`. After finalize, sorted position encodes the offset bits (tree-walk order within bin).

3. **Get item**: Binary search over cumulative bin counts to find the bin. Index lookup to find the entry. Reconstruct full key from `bin_id` (→ low bits) + `sorted_position` (→ offset bits) + `entry` (→ trailing bits).

### DRAM Bandwidth

At e=30, p=32, lb=20 (1024 bins):

| Phase | Operation | B/key |
|-------|-----------|-------|
| Process | Read key + write entry | ~4.25 |
| Finalize | Read entry + write index | ~2.75 |
| **Total** | | **~7 B/key** |

vs radix sort at **32 B/key** inherent — **4.6× less bandwidth**.

### Optimal lb Selection

The `lb` parameter controls the trade-off between bin count and entry size:

| Rule | Bins | Use case |
|------|------|----------|
| `lb = e - 10` | 1024 | More cached counter levels |
| `lb = e - 8` | 256 | Best bandwidth ratio vs radix |
| `lb = e - 6` | 64 | Fewest bins, simplest scatter |

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
    lb=None,        # log2(bin size), default: e - 8
    n_batches=4,    # processing batches (for streaming)
)
```

### Result object

```python
result.get_item(position)    # O(log bins) point query
result[i]                    # same via __getitem__
result[10:20]                # slice access
len(result)                  # total number of keys
result.reconstruct_all()     # all keys in sorted order
```

### Internal arrays (for advanced use)

```python
result.sbatch_mem       # entry array (per-bin regions)
result.index_array      # sorted_pos → arrival_pos mapping
result.bin_counts       # entries per bin
result.bin_cumulative   # cumulative bin starts
result.hist             # per-bin histogram
result.ln               # tree depth
result.lb               # log2(bin size)
result.entry_bits       # bits per entry (lb + trailing)
result.n_bins           # number of bins
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
