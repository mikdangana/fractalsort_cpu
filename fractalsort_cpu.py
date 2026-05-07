"""fractalsort_cpu: Fractal sort for CPU.

Architecture:
  A p-bit key is decomposed into three parts:
    bits 0..ln-lb-1    -> bin_id (tree-ordered via bit-reverse, ln-lb bits)
    bits ln-lb..ln-1   -> offset within bin (lb bits, encoded by sorted position)
    bits ln..p-1       -> trailing bits (p-ln bits, stored in entry)

  Process phase: route each key to one of 2^(ln-lb) bins in tree order.
    Store only (tree_offset | trailing) per entry. Direct scatter with
    bin write pointers in L1 cache. DRAM: read key (4B) + write entry.

  Finalize phase: radix-sort within each bin by entry value, producing
    an index array that maps sorted_position -> arrival_position.

  Get item: tree walk over segment tree of bin counts to find bin,
    index lookup, reconstruct full key from bin_id + sorted_position
    + trailing bits.
"""
import numpy as np
from numba import njit


# === LUT bit reversal ===

@njit(nogil=True)
def _build_reverse_lut():
    """Build 16-bit reversal lookup table."""
    table = np.empty(65536, dtype=np.uint16)
    for i in range(65536):
        r = 0; v = i
        for b in range(16):
            r = (r << 1) | (v & 1); v >>= 1
        table[i] = np.uint16(r)
    return table


@njit(nogil=True, fastmath=True)
def _bit_reverse_lut32(val, nbits, lut):
    """Bit-reverse lowest nbits of val using LUT (up to 32 bits)."""
    if nbits <= 16:
        return np.int64(lut[int(val & 0xFFFF)]) >> np.int64(16 - nbits)
    else:
        lo = np.int64(lut[int(val & 0xFFFF)]) << np.int64(nbits - 16)
        hi = np.int64(lut[int((val >> 16) & 0xFFFF)]) >> np.int64(32 - nbits)
        return lo | hi


# === Core kernels ===

@njit(nogil=True, fastmath=True)
def _precount_bins(keys, ln_minus_lb, lut):
    """Count how many keys fall in each bin (pre-pass for allocation)."""
    n_bins = np.int64(1) << ln_minus_lb
    mask = np.int64(n_bins - 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(keys.size):
        bid = _bit_reverse_lut32(np.int64(keys[i]) & mask, ln_minus_lb, lut)
        counts[bid] += 1
    return counts


@njit(nogil=True, fastmath=True)
def _process_batch(keys, hist, ln_minus_lb, lb, p, ln,
                   sbatch_mem, bin_write_pos, lut):
    """Single-pass direct scatter. bin_id is a scalar (register), not stored.

    Entry = (tree_reversed_offset << trailing_bits) | trailing.
    DRAM traffic: read key (4B) + write entry (4B) = 8 B/key.
    """
    nk = keys.size
    n_bins = np.int64(1) << ln_minus_lb
    bin_mask = np.int64(n_bins - 1)
    offset_mask = np.uint32((1 << lb) - 1)
    trailing_bits = p - ln
    trailing_mask = np.uint32((1 << trailing_bits) - 1) if trailing_bits > 0 else np.uint32(0)

    for i in range(nk):
        ki = keys[i]

        # Bin id: scalar in register, not stored to memory
        bid = _bit_reverse_lut32(np.int64(ki) & bin_mask, ln_minus_lb, lut)
        hist[bid] += 1

        # Entry: tree-reversed offset | trailing
        raw_offset = np.int64((ki >> np.uint32(ln_minus_lb)) & offset_mask)
        tree_offset = np.uint32(_bit_reverse_lut32(raw_offset, lb, lut))
        if trailing_bits > 0:
            trailing = (ki >> np.uint32(ln)) & trailing_mask
            entry = (tree_offset << np.uint32(trailing_bits)) | trailing
        else:
            entry = tree_offset

        # Direct scatter to per-bin region
        wp = bin_write_pos[bid]
        sbatch_mem[wp] = entry
        bin_write_pos[bid] = wp + np.int64(1)


@njit(nogil=True, fastmath=True)
def _finalize(sbatch_mem, bin_counts, n_bins, entry_bits):
    """Radix-sort within each bin by entry value, produce index array.

    index_array[bin_offset + sorted_pos] = arrival_position_within_bin.
    """
    total = np.int64(0)
    for b in range(n_bins):
        total += np.int64(bin_counts[b])

    index_array = np.empty(total, dtype=np.int32)
    offset = np.int64(0)
    n_passes = (entry_bits + 7) // 8

    for b in range(n_bins):
        count = np.int64(bin_counts[b])
        if count == 0:
            continue

        # Initialize permutation
        perm = np.empty(count, dtype=np.int32)
        for i in range(count):
            perm[i] = np.int32(i)
        temp = np.empty(count, dtype=np.int32)

        # Radix sort by 8-bit digits of entry value
        for pass_idx in range(n_passes):
            shift = np.uint32(pass_idx * 8)
            counts = np.zeros(256, dtype=np.int64)
            for i in range(count):
                v = (sbatch_mem[offset + np.int64(perm[i])] >> shift) & np.uint32(0xFF)
                counts[v] += 1
            offsets = np.zeros(257, dtype=np.int64)
            for v in range(256):
                offsets[v + 1] = offsets[v] + counts[v]
            for i in range(count):
                v = (sbatch_mem[offset + np.int64(perm[i])] >> shift) & np.uint32(0xFF)
                temp[offsets[v]] = perm[i]
                offsets[v] += 1
            for i in range(count):
                perm[i] = temp[i]

        # Write result: index_array[sorted_pos] = arrival_pos
        for i in range(count):
            index_array[offset + np.int64(i)] = perm[i]

        offset += count

    return index_array


@njit(nogil=True)
def _build_seg_tree(bin_counts, n_bins):
    """Build segment tree from bin counts. 1-indexed: leaves at n_bins..2*n_bins-1."""
    tree = np.zeros(2 * n_bins, dtype=np.int64)
    for i in range(n_bins):
        tree[n_bins + i] = bin_counts[i]
    for i in range(n_bins - 1, 0, -1):
        tree[i] = tree[2 * i] + tree[2 * i + 1]
    return tree


@njit(nogil=True, fastmath=True)
def _get_item(sbatch_mem, index_array, seg_tree, bin_cumulative, n_bins,
              ln, lb, p, ln_minus_lb, lut, position):
    """Get the key at a given sorted position.

    1. Tree walk over segment tree to find bin (O(log n_bins), cache-friendly)
    2. Index lookup for arrival position
    3. Extract trailing bits from entry
    4. Reconstruct full key from bin_id + offset + trailing
    """
    pos = np.int64(position)

    # Tree walk to find bin
    node = np.int64(1)
    while node < n_bins:
        left = 2 * node
        left_count = seg_tree[left]
        if pos < left_count:
            node = left
        else:
            pos -= left_count
            node = left + 1
    bin_id = node - np.int64(n_bins)
    bin_pos = pos
    bin_offset = bin_cumulative[bin_id]

    # Index lookup
    arrival_pos = np.int64(index_array[bin_offset + bin_pos])
    entry = sbatch_mem[bin_offset + arrival_pos]

    # Extract trailing and tree_offset from entry
    trailing_bits = p - ln
    if trailing_bits > 0:
        trailing_mask = np.uint32((1 << trailing_bits) - 1)
        trailing = entry & trailing_mask
        tree_offset = np.int64(entry >> np.uint32(trailing_bits))
    else:
        trailing = np.uint32(0)
        tree_offset = np.int64(entry)

    # Recover offset bits (reverse tree ordering)
    offset_val = np.uint32(_bit_reverse_lut32(tree_offset, lb, lut))

    # Recover key low bits from bin_id (reverse tree ordering)
    key_low = np.uint32(_bit_reverse_lut32(bin_id, ln_minus_lb, lut))

    # Reconstruct full key
    key = key_low | (offset_val << np.uint32(ln_minus_lb))
    if trailing_bits > 0:
        key |= trailing << np.uint32(ln)
    return key


@njit(nogil=True, fastmath=True)
def _reconstruct_all(sbatch_mem, index_array, bin_counts, bin_cumulative,
                     n_bins, ln, lb, p, ln_minus_lb, lut):
    """Reconstruct all keys in sorted (tree-walk) order."""
    total = np.int64(0)
    for b in range(n_bins):
        total += np.int64(bin_counts[b])

    keys_out = np.empty(total, dtype=np.uint32)
    trailing_bits = p - ln
    trailing_mask = np.uint32((1 << trailing_bits) - 1) if trailing_bits > 0 else np.uint32(0)
    offset = np.int64(0)
    out_idx = np.int64(0)

    for b in range(n_bins):
        count = np.int64(bin_counts[b])
        if count == 0:
            continue

        # Recover bits 0..ln-lb-1 from bin_id
        key_low = np.uint32(_bit_reverse_lut32(np.int64(b), ln_minus_lb, lut))

        for sp in range(count):
            arrival_pos = np.int64(index_array[offset + sp])
            entry = sbatch_mem[offset + arrival_pos]

            if trailing_bits > 0:
                trailing = entry & trailing_mask
                tree_offset = np.int64(entry >> np.uint32(trailing_bits))
            else:
                trailing = np.uint32(0)
                tree_offset = np.int64(entry)

            offset_val = np.uint32(_bit_reverse_lut32(tree_offset, lb, lut))

            key = key_low | (offset_val << np.uint32(ln_minus_lb))
            if trailing_bits > 0:
                key |= trailing << np.uint32(ln)
            keys_out[out_idx] = key
            out_idx += 1

        offset += count

    return keys_out


# === High-level API ===

class FractalSortResult:
    """Result of fractal sort. Supports indexed access to sorted keys."""

    def __init__(self, sbatch_mem, index_array, bin_counts, bin_cumulative,
                 hist, ln, lb, p, n_bins, lut, seg_tree):
        self.sbatch_mem = sbatch_mem
        self.index_array = index_array
        self.bin_counts = bin_counts
        self.bin_cumulative = bin_cumulative
        self.hist = hist
        self.seg_tree = seg_tree
        self.ln = ln
        self.lb = lb
        self.p = p
        self.n_bins = n_bins
        self.ln_minus_lb = ln - lb
        self.entry_bits = lb + (p - ln)
        self.lut = lut
        self._n = int(np.sum(bin_counts))

    def __len__(self):
        return self._n

    def __getitem__(self, position):
        if isinstance(position, slice):
            start, stop, step = position.indices(self._n)
            return np.array([self.get_item(i) for i in range(start, stop, step)],
                            dtype=np.uint32)
        if position < 0:
            position += self._n
        if position < 0 or position >= self._n:
            raise IndexError(f"position {position} out of range [0, {self._n})")
        return self.get_item(position)

    def get_item(self, position):
        """Get the key at a given sorted position. O(log n_bins) via tree walk."""
        return _get_item(self.sbatch_mem, self.index_array, self.seg_tree,
                         self.bin_cumulative, self.n_bins, self.ln, self.lb,
                         self.p, self.ln_minus_lb, self.lut, np.int64(position))

    def reconstruct_all(self):
        """Reconstruct all keys in sorted order. Returns uint32 array."""
        return _reconstruct_all(self.sbatch_mem, self.index_array,
                                self.bin_counts, self.bin_cumulative,
                                self.n_bins, self.ln, self.lb, self.p,
                                self.ln_minus_lb, self.lut)


# Global LUT (built once on import)
_LUT = _build_reverse_lut()


def fractalsort(keys, p=32, lb=None, n_batches=4):
    """Sort keys using fractal sort.

    Args:
        keys: uint32 array of p-bit keys
        p: key precision in bits (default 32)
        lb: log2(bin size). Controls bins = 2^(e-lb). Default: e - 8.
        n_batches: number of processing batches

    Returns:
        FractalSortResult with indexed access to sorted keys.
    """
    n = keys.size
    e = int(np.ceil(np.log2(n))) if n > 1 else 1
    ln = e

    if lb is None:
        lb = max(1, e - 8)
    lb = min(lb, ln - 1)
    ln_minus_lb = ln - lb
    n_bins = 1 << ln_minus_lb
    entry_bits = lb + (p - ln)
    batch_size = (n + n_batches - 1) // n_batches

    lut = _LUT

    # Pre-count bins for contiguous allocation
    all_bc = _precount_bins(keys, ln_minus_lb, lut)
    bin_starts = np.zeros(n_bins, dtype=np.int64)
    for i in range(1, n_bins):
        bin_starts[i] = bin_starts[i - 1] + all_bc[i - 1]

    # Allocate output arrays
    sbatch_mem = np.empty(n, dtype=np.uint32)
    hist = np.zeros(n_bins, dtype=np.int64)
    bin_write_pos = bin_starts.copy()

    # Process batches (single-pass direct scatter)
    for b in range(n_batches):
        i0 = b * batch_size
        i1 = min(i0 + batch_size, n)
        if i0 >= n:
            break
        _process_batch(keys[i0:i1], hist, ln_minus_lb, lb, p, ln,
                       sbatch_mem, bin_write_pos, lut)

    # Bin counts from write positions
    bin_counts = np.empty(n_bins, dtype=np.int64)
    for i in range(n_bins):
        bin_counts[i] = bin_write_pos[i] - bin_starts[i]

    # Finalize: sort within bins, produce index
    index_array = _finalize(sbatch_mem, bin_counts, n_bins, entry_bits)

    # Cumulative bin starts (for index offset lookup)
    bin_cumulative = np.zeros(n_bins, dtype=np.int64)
    for i in range(1, n_bins):
        bin_cumulative[i] = bin_cumulative[i - 1] + bin_counts[i - 1]

    # Segment tree for O(log n_bins) tree-walk get_item
    seg_tree = _build_seg_tree(bin_counts, n_bins)

    return FractalSortResult(
        sbatch_mem=sbatch_mem,
        index_array=index_array,
        bin_counts=bin_counts,
        bin_cumulative=bin_cumulative,
        hist=hist,
        ln=ln, lb=lb, p=p, n_bins=n_bins, lut=lut,
        seg_tree=seg_tree,
    )
