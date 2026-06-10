"""Test frmw_io: verify counter correctness and key reconstruction."""
import numpy as np
import math
import sys

e = int(sys.argv[1]) if len(sys.argv) > 1 else 14
cache_mb = int(sys.argv[2]) if len(sys.argv) > 2 else 8
p = 32
lb = 10  # b = 1024 bins per group
n_batches = 4
wc_margin = 2

n = 1 << e
ln = e
batch_size = n // n_batches

print(f"n=2^{e}={n:,}, p={p}, lb={lb}, n_batches={n_batches}")

# Build layout for levels 0..lc-1 only
n_levels_full = ln + 1
widths_full = np.array([max(2, e + 1 - l + wc_margin) for l in range(n_levels_full)], dtype=np.int64)
per_word_full = np.array([64 // w for w in widths_full], dtype=np.int64)
n_bins_arr = np.array([1 << l for l in range(n_levels_full)], dtype=np.int64)
n_words_full = np.array([(nb + pw - 1) // pw for nb, pw in zip(n_bins_arr, per_word_full)], dtype=np.int64)

# Determine lc
cache_bytes = cache_mb * 1024 * 1024
cum = 0
lc = 0
for l in range(n_levels_full):
    cum += int(n_words_full[l]) * 8
    if cum > cache_bytes // 2:
        break
    lc = l
lc = min(lc, ln - lb)  # ensure lc <= ln - lb
if lc < 1:
    lc = 1

# Build layout for levels 0..lc-1
n_levels_cached = lc
widths = widths_full[:n_levels_cached].copy()
per_word = per_word_full[:n_levels_cached].copy()
word_offsets = np.zeros(n_levels_cached, dtype=np.int64)
for l in range(1, n_levels_cached):
    word_offsets[l] = word_offsets[l - 1] + n_words_full[l - 1]
total_words = int(word_offsets[-1] + n_words_full[n_levels_cached - 1]) if n_levels_cached > 0 else 1

ln_minus_lb = ln - lb
n_io_bins = 1 << ln_minus_lb
routing_bits = ln - lb - lc
trailing_bits = p - ln
entry_bits = lb + trailing_bits  # offset bits + trailing bits

print(f"lc={lc}, ln={ln}, lb={lb}")
print(f"n_io_bins=2^{ln_minus_lb}={n_io_bins:,}")
print(f"entry_bits={entry_bits} (offset={lb} + trailing={trailing_bits})")
print(f"counter levels: 0..{lc-1}, total_words={total_words}")

from fractalsort_core.frmw_io import frmw_io_process, frmw_io_finalize, frmw_io_reconstruct, _bit_reverse

# Reference: naive per-key counter update for levels 0..lc-1
from numba import njit


@njit(nogil=True, fastmath=True)
def ref_counter_update(keys, cnts, widths, per_word, word_offsets, n_levels_cached):
    for i in range(keys.size):
        ki = np.uint64(keys[i])
        bin_idx = np.int64(0)
        for l in range(n_levels_cached):
            w = widths[l]
            pw = per_word[l]
            word_idx = word_offsets[l] + bin_idx // pw
            bit_pos = np.uint64((bin_idx % pw) * w)
            cnts[np.intp(word_idx)] += np.uint64(1) << bit_pos
            bin_idx = bin_idx | (np.int64((ki >> np.uint64(l)) & np.uint64(1)) << l)


@njit(nogil=True, fastmath=True)
def precount_bins(keys, ln_minus_lb):
    """Count how many keys land in each bin across all keys."""
    n_bins = np.int64(1) << ln_minus_lb
    mask = np.int64((1 << ln_minus_lb) - 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(keys.size):
        bid = _bit_reverse(np.int64(keys[i]) & mask, ln_minus_lb)
        counts[bid] += 1
    return counts


np.random.seed(42)
keys = np.random.randint(0, 1 << p, size=n, dtype=np.uint32)

# --- Compile ---
print("\nCompiling...")
small = keys[:min(256, batch_size)].copy()
cnts_tmp = np.zeros(total_words, dtype=np.uint64)
ref_counter_update(small, cnts_tmp, widths, per_word, word_offsets, n_levels_cached)

_ = precount_bins(small, ln_minus_lb)

cnts_tmp[:] = 0
sb_tmp = np.empty(n, dtype=np.uint32)
bc_tmp = np.zeros(n_io_bins, dtype=np.int32)
wp_tmp = np.zeros(n_io_bins, dtype=np.int64)
# Set up bin_starts for small test
small_counts = precount_bins(small, ln_minus_lb)
starts = np.zeros(n_io_bins, dtype=np.int64)
for i in range(1, n_io_bins):
    starts[i] = starts[i-1] + small_counts[i-1]
wp_tmp[:] = starts
frmw_io_process(small, cnts_tmp, widths, per_word, word_offsets, lc, ln, lb, p,
                sb_tmp, bc_tmp, wp_tmp)

# Finalize compile
total_small = int(np.sum(bc_tmp))
if total_small > 0:
    _ = frmw_io_finalize(sb_tmp[:total_small], bc_tmp, n_io_bins, entry_bits)
    _ = frmw_io_reconstruct(sb_tmp[:total_small], _, bc_tmp, n_io_bins, ln, lb, lc, p, ln_minus_lb)

print("Done.\n")

# --- Correctness test ---
print("=== Correctness Test ===")

# Pre-count all bins to allocate contiguous per-bin regions
all_bin_counts = precount_bins(keys, ln_minus_lb)
bin_starts = np.zeros(n_io_bins, dtype=np.int64)
for i in range(1, n_io_bins):
    bin_starts[i] = bin_starts[i-1] + all_bin_counts[i-1]

# Reference counters
cnts_ref = np.zeros(total_words, dtype=np.uint64)
for b in range(n_batches):
    i0, i1 = b * batch_size, (b + 1) * batch_size
    ref_counter_update(keys[i0:i1], cnts_ref, widths, per_word, word_offsets, n_levels_cached)

# frmw_io counters + sbatch
cnts_io = np.zeros(total_words, dtype=np.uint64)
sbatch_mem = np.empty(n, dtype=np.uint32)
sbatch_bin_counts = np.zeros(n_io_bins, dtype=np.int32)
bin_write_pos = bin_starts.copy()  # per-bin write cursors

for b in range(n_batches):
    i0, i1 = b * batch_size, (b + 1) * batch_size
    frmw_io_process(keys[i0:i1], cnts_io, widths, per_word, word_offsets, lc, ln, lb, p,
                    sbatch_mem, sbatch_bin_counts, bin_write_pos)

total_entries = int(np.sum(sbatch_bin_counts))
print(f"Total entries written: {total_entries:,} (expected {n:,})")

# Check counters match
if np.array_equal(cnts_ref, cnts_io):
    print("Counters (levels 0..lc-1): PASS")
else:
    d = np.where(cnts_ref != cnts_io)[0]
    print(f"Counters: FAIL at {len(d)} words (first 5: {d[:5]})")

# Check bin counts
total_binned = np.sum(sbatch_bin_counts)
print(f"Bin counts sum: {total_binned:,} (expected {n:,}) {'PASS' if total_binned == n else 'FAIL'}")

# Verify per-bin write positions match expected
expected_end = bin_starts + all_bin_counts.astype(np.int64)
if np.array_equal(bin_write_pos, expected_end):
    print("Per-bin write positions: PASS")
else:
    print("Per-bin write positions: FAIL")

# Finalize: sort within bins, produce index
print("\nRunning finalize (sort within bins)...")
index_array = frmw_io_finalize(sbatch_mem[:total_entries], sbatch_bin_counts, n_io_bins, entry_bits)
print(f"Index array size: {len(index_array):,}")

# Reconstruct keys
print("Reconstructing keys...")
recon_keys = frmw_io_reconstruct(sbatch_mem[:total_entries], index_array, sbatch_bin_counts,
                                  n_io_bins, ln, lb, lc, p, ln_minus_lb)
print(f"Reconstructed {len(recon_keys):,} keys")

# Verify: every original key appears in reconstruction (as a multiset)
orig_sorted = np.sort(keys)
recon_sorted = np.sort(recon_keys)

if np.array_equal(orig_sorted, recon_sorted):
    print("Key reconstruction: PASS (all keys recovered)")
else:
    n_match = np.sum(orig_sorted == recon_sorted)
    print(f"Key reconstruction: FAIL ({n_match}/{n} match after sorting)")
    diff_mask = orig_sorted != recon_sorted
    diff_idx = np.where(diff_mask)[0][:5]
    for di in diff_idx:
        print(f"  pos {di}: orig=0x{orig_sorted[di]:08x} recon=0x{recon_sorted[di]:08x}")

# Memory analysis
entry_bytes = total_entries * 4  # uint32 storage (could be packed tighter)
index_bytes = total_entries * 2  # uint16
counter_bytes = total_words * 8
bin_count_bytes = n_io_bins * 4
total_mem = entry_bytes + index_bytes + counter_bytes + bin_count_bytes
key_bytes = n * 4

print(f"\n=== Memory Analysis ===")
print(f"Input keys:     {key_bytes:>12,} bytes ({key_bytes/n:.1f} B/key)")
print(f"Sbatch entries: {entry_bytes:>12,} bytes ({entry_bytes/n:.1f} B/key) [{entry_bits} bits needed, stored as uint32]")
print(f"Index array:    {index_bytes:>12,} bytes ({index_bytes/n:.1f} B/key) [{lb} bits needed, stored as uint16]")
print(f"Counters:       {counter_bytes:>12,} bytes ({counter_bytes/n:.1f} B/key) [levels 0..{lc-1}]")
print(f"Bin counts:     {bin_count_bytes:>12,} bytes ({bin_count_bytes/n:.1f} B/key)")
print(f"TOTAL output:   {total_mem:>12,} bytes ({total_mem/n:.1f} B/key)")
if total_mem > 0:
    print(f"Compression vs raw keys: {key_bytes/total_mem:.2f}x")

# Packed estimate
packed_entry_bytes = math.ceil(total_entries * entry_bits / 8)
packed_index_bytes = math.ceil(total_entries * lb / 8)
packed_total = packed_entry_bytes + packed_index_bytes + counter_bytes + bin_count_bytes
print(f"\nPacked estimate: {packed_total:>12,} bytes ({packed_total/n:.1f} B/key)")
if packed_total > 0:
    print(f"Packed compression: {key_bytes/packed_total:.2f}x")
