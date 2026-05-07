"""frmw_io_fast: Optimized frmw_io with LUT bit-reversal and bottom-up histogram.

Replaces:
  - Per-key bit_reverse loop → 16-bit LUT
  - Per-key counter update (lc iterations) → single histogram build + reduction
"""
import numpy as np
from numba import njit, prange


@njit(nogil=True, cache=True)
def _build_reverse_lut():
    """Build 16-bit reversal lookup table."""
    table = np.empty(65536, dtype=np.uint16)
    for i in range(65536):
        r = 0
        v = i
        for b in range(16):
            r = (r << 1) | (v & 1)
            v >>= 1
        table[i] = np.uint16(r)
    return table


@njit(nogil=True, fastmath=True, cache=True)
def _bit_reverse_lut(val, nbits, lut):
    """Fast bit reverse using LUT. nbits <= 16."""
    return np.int64(lut[int(val & 0xFFFF)]) >> np.int64(16 - nbits)


@njit(nogil=True, fastmath=True, cache=True)
def _bit_reverse_lut32(val, nbits, lut):
    """Fast bit reverse for up to 32 bits using LUT."""
    if nbits <= 16:
        return np.int64(lut[int(val & 0xFFFF)]) >> np.int64(16 - nbits)
    else:
        lo = np.int64(lut[int(val & 0xFFFF)]) << np.int64(nbits - 16)
        hi = np.int64(lut[int((val >> 16) & 0xFFFF)]) >> np.int64(32 - nbits)
        return lo | hi


@njit(nogil=True, fastmath=True, cache=True)
def frmw_io_fast_process(keys, hist, ln_minus_lb, lb, p, ln,
                         sbatch_mem, bin_write_pos, sbatch_bin_counts, lut):
    """Optimized frmw_io process: LUT bit-reverse + histogram for counters.

    Args:
        keys: input batch
        hist: histogram array [2^(ln_minus_lb)] for counter tree (accumulated)
        ln_minus_lb: bits for bin_id
        lb: bits for within-bin offset
        p: key precision
        ln: tree depth
        sbatch_mem: output entry array (per-bin regions)
        bin_write_pos: per-bin write cursors
        sbatch_bin_counts: per-bin counts (accumulated)
        lut: 16-bit reversal LUT
    """
    nk = keys.size
    n_bins = np.int64(1) << ln_minus_lb
    bin_mask = np.int64(n_bins - 1)
    offset_mask = np.uint32((1 << lb) - 1)
    trailing_bits = p - ln
    trailing_mask = np.uint32((1 << trailing_bits) - 1) if trailing_bits > 0 else np.uint32(0)

    # Per-batch local bin counts for count-sort
    local_counts = np.zeros(n_bins, dtype=np.int32)
    bin_ids = np.empty(nk, dtype=np.int32)
    entries = np.empty(nk, dtype=np.uint32)

    for i in range(nk):
        ki = keys[i]

        # Bin_id: bit_reverse of lower (ln-lb) key bits
        raw_low = np.int64(ki) & bin_mask
        bid = np.int32(_bit_reverse_lut32(raw_low, ln_minus_lb, lut))
        bin_ids[i] = bid
        local_counts[bid] += 1

        # Histogram for counter tree (indexed by bin_id = tree-ordered bits 0..ln-lb-1)
        hist[bid] += 1

        # Entry: tree-reversed offset | trailing
        raw_offset = np.int64((ki >> np.uint32(ln_minus_lb)) & offset_mask)
        tree_offset = np.uint32(_bit_reverse_lut(raw_offset, lb, lut))
        if trailing_bits > 0:
            trailing = (ki >> np.uint32(ln)) & trailing_mask
            entry = (tree_offset << np.uint32(trailing_bits)) | trailing
        else:
            entry = tree_offset
        entries[i] = entry

    # Count-sort entries by bin_id within this batch, then scatter to per-bin regions
    # Build offsets for local sort
    offsets = np.zeros(n_bins + 1, dtype=np.int64)
    for b in range(n_bins):
        offsets[b + 1] = offsets[b] + np.int64(local_counts[b])

    # Scatter to sorted buffer
    sorted_entries = np.empty(nk, dtype=np.uint32)
    pos = offsets[:n_bins].copy()
    for i in range(nk):
        b = np.int64(bin_ids[i])
        sorted_entries[pos[b]] = entries[i]
        pos[b] += 1

    # Write sorted entries to per-bin regions in sbatch_mem
    for b in range(n_bins):
        wp = bin_write_pos[b]
        start = offsets[b]
        count = local_counts[b]
        for j in range(count):
            sbatch_mem[wp + j] = sorted_entries[start + j]
        bin_write_pos[b] = wp + np.int64(count)
        sbatch_bin_counts[b] += count


@njit(nogil=True, fastmath=True, cache=True)
def frmw_io_fast_finalize(sbatch_mem, sbatch_bin_counts, n_bins, entry_bits):
    """Sort within each bin (counting sort for small entry_bits, else insertion)."""
    total = np.int64(0)
    for b in range(n_bins):
        total += np.int64(sbatch_bin_counts[b])

    index_array = np.empty(total, dtype=np.uint16)
    offset = np.int64(0)

    use_counting = entry_bits <= 16  # counting sort feasible up to 2^16 buckets

    for b in range(n_bins):
        count = np.int64(sbatch_bin_counts[b])
        if count == 0:
            continue

        if use_counting:
            n_vals = np.int64(1) << entry_bits
            val_counts = np.zeros(n_vals, dtype=np.int32)
            for i in range(count):
                val_counts[sbatch_mem[offset + i]] += 1

            val_offsets = np.zeros(n_vals + 1, dtype=np.int64)
            for v in range(n_vals):
                val_offsets[v + 1] = val_offsets[v] + np.int64(val_counts[v])

            rank = np.zeros(n_vals, dtype=np.int32)
            for i in range(count):
                v = sbatch_mem[offset + i]
                sorted_pos = val_offsets[v] + np.int64(rank[v])
                index_array[offset + sorted_pos] = np.uint16(i)
                rank[v] += 1
        else:
            # Insertion sort fallback
            perm = np.empty(count, dtype=np.int32)
            for i in range(count):
                perm[i] = np.int32(i)
            for i in range(1, count):
                key_val = sbatch_mem[offset + perm[i]]
                tmp = perm[i]
                j = i - 1
                while j >= 0 and sbatch_mem[offset + perm[j]] > key_val:
                    perm[j + 1] = perm[j]
                    j -= 1
                perm[j + 1] = tmp
            for i in range(count):
                index_array[offset + i] = np.uint16(perm[i])

        offset += count

    return index_array


@njit(nogil=True, fastmath=True, cache=True)
def build_counters_from_hist(hist, cnts, widths, per_word, word_offsets, lc, n_bins):
    """Build packed counter tree levels 0..lc-1 from the histogram via bottom-up reduction.

    hist: array of size n_bins (= 2^(ln-lb)), where hist[b] = count of keys in bin b.
    Bin b in tree order: MSBs = bits 0..lc-1. So level-l count for bin_l =
    sum of hist entries whose top l bits of bin_id = bin_l.
    """
    # Level lc-1 is directly from histogram (just need to aggregate groups)
    # Actually: hist has 2^(ln-lb) entries. Level l has 2^l bins.
    # Level l bin b = all hist entries whose top l bits = b
    # = sum of hist[b * 2^(ln-lb-l) : (b+1) * 2^(ln-lb-l)]

    ln_minus_lb = 0
    tmp = n_bins
    while tmp > 1:
        ln_minus_lb += 1
        tmp >>= 1

    # Start from the finest level we can (ln-lb entries), reduce upward
    # Work array = copy of hist
    work = np.empty(n_bins, dtype=np.int64)
    for i in range(n_bins):
        work[i] = np.int64(hist[i])

    # Pack each level from bottom to top
    n_entries = n_bins
    for l in range(ln_minus_lb - 1, -1, -1):
        if l < lc:
            # Pack this level into cnts
            w = widths[l]
            pw = per_word[l]
            wo = word_offsets[l]
            n_words_l = (n_entries + pw - 1) // pw
            for wi in range(n_words_l):
                packed = np.uint64(0)
                base = wi * pw
                for s in range(pw):
                    b = base + s
                    if b >= n_entries:
                        break
                    c = work[b]
                    if c > 0:
                        packed += np.uint64(c) << np.uint64(s * w)
                if packed != 0:
                    cnts[np.intp(wo + wi)] += packed

        # Reduce: merge pairs
        half = n_entries >> 1
        for b in range(half):
            work[b] = work[2 * b] + work[2 * b + 1]
        n_entries = half

    # Level 0 (root): total count
    if lc > 0:
        cnts[np.intp(word_offsets[0])] += np.uint64(work[0])
