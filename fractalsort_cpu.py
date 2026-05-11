"""fractalsort_cpu: Fractal sort for CPU.

Architecture:
  A p-bit key is decomposed into two parts:
    top ln-lb bits     -> bin_id (determines which bin, ascending order)
    bottom entry_bits  -> entry (lb + (p-ln) bits, stored per key)

  entry_bits = lb + (p - ln), where ln = ceil(log2(n)).

  Process phase: route each key to one of 2^(ln-lb) bins by its MSBs.
    Store entry = key & ((1 << entry_bits) - 1). Per-batch sort within
    each bin keeps entries in sorted runs (one per batch).

  No merge/index phase — sorted runs stay concatenated.

  Get item: segment tree walk to find bin, then k-way selection
    across sorted runs via binary search.

  Reconstruct all: k-way merge of sorted runs at read time.
"""
import numpy as np
import time
from numba import njit, prange


# === Bandwidth instrumentation ===

@njit(nogil=True, fastmath=True)
def _stream_read(arr):
    """Stream read: sum all elements (forces DRAM read). Returns checksum."""
    s = np.uint64(0)
    for i in range(arr.size):
        s += np.uint64(arr[i])
    return s


@njit(nogil=True, fastmath=True)
def _stream_copy(src, dst):
    """Stream copy: read src, write dst. Known cost = 8 B/key for uint32."""
    for i in range(src.size):
        dst[i] = src[i]


def calibrate_bandwidth(n=1 << 27, reps=5):
    """Measure peak DRAM bandwidth via stream read and copy.

    Returns dict with read_GBs, copy_GBs, and helper to estimate B/key.
    """
    arr = np.random.randint(0, 2**32, size=n, dtype=np.uint32)
    dst = np.empty_like(arr)

    # Warmup
    _stream_read(arr)
    _stream_copy(arr, dst)

    # Stream read: 4 B/key
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _stream_read(arr)
        dt = time.perf_counter() - t0
        times.append(dt)
    read_time = min(times)
    read_GBs = (n * 4) / read_time / 1e9

    # Stream copy: 8 B/key (4 read + 4 write)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _stream_copy(arr, dst)
        dt = time.perf_counter() - t0
        times.append(dt)
    copy_time = min(times)
    copy_GBs = (n * 8) / copy_time / 1e9

    return {
        'n': n,
        'read_GBs': read_GBs,
        'copy_GBs': copy_GBs,
        'read_time': read_time,
        'copy_time': copy_time,
    }


def estimate_bw(sort_time, n, cal):
    """Estimate B/key from sort time using calibration data.

    Uses copy bandwidth (read+write) as the reference:
      B/key = sort_time / n * peak_bandwidth
    """
    peak_bw = cal['copy_GBs'] * 1e9  # bytes/sec
    total_bytes = sort_time * peak_bw
    return total_bytes / n


# === Core kernels ===

@njit(nogil=True, fastmath=True)
def _precount_bins(keys, entry_bits, n_bins):
    """Count how many keys fall in each bin (pre-pass for allocation)."""
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(keys.size):
        bid = np.int64(keys[i] >> np.uint32(entry_bits))
        if bid < n_bins:
            counts[bid] += 1
    return counts


@njit(nogil=True, fastmath=True)
def _process_batch(keys, entry_bits, n_bins, sbatch_mem, bin_write_pos):
    """Single-pass direct scatter. bin_id from MSBs.

    Entry = bottom entry_bits of key.
    """
    nk = keys.size
    entry_mask = np.uint32((1 << entry_bits) - 1)

    for i in range(nk):
        ki = keys[i]
        bid = np.int64(ki >> np.uint32(entry_bits))
        if bid >= n_bins:
            bid = np.int64(n_bins - 1)
        entry = ki & entry_mask
        wp = bin_write_pos[bid]
        sbatch_mem[wp] = entry
        bin_write_pos[bid] = wp + np.int64(1)


@njit(nogil=True, fastmath=True, parallel=True)
def _sort_batch_in_bins(sbatch_mem, batch_starts, batch_ends, n_bins,
                        max_passes):
    """Sort each bin's entries in-place (parallel). Only sort max_passes bytes."""
    for b in prange(n_bins):
        start = batch_starts[b]
        end = batch_ends[b]
        count = end - start
        if count <= 1:
            continue

        if count <= 64:
            # Insertion sort for small chunks
            for i in range(start + 1, end):
                val = sbatch_mem[i]
                j = i - 1
                while j >= start and sbatch_mem[j] > val:
                    sbatch_mem[j + 1] = sbatch_mem[j]
                    j -= 1
                sbatch_mem[j + 1] = val
        else:
            # Radix sort — only ceil(entry_bits/8) passes
            n = int(count)
            buf = np.empty(n, dtype=np.uint32)
            src = np.empty(n, dtype=np.uint32)
            for i in range(n):
                src[i] = sbatch_mem[start + i]

            for pass_idx in range(max_passes):
                shift = np.uint32(pass_idx * 8)
                counts = np.zeros(256, dtype=np.int64)
                for i in range(n):
                    v = (src[i] >> shift) & np.uint32(0xFF)
                    counts[v] += 1
                offsets = np.zeros(256, dtype=np.int64)
                for v in range(1, 256):
                    offsets[v] = offsets[v - 1] + counts[v - 1]
                for i in range(n):
                    v = (src[i] >> shift) & np.uint32(0xFF)
                    buf[offsets[v]] = src[i]
                    offsets[v] += 1
                src, buf = buf, src

            for i in range(n):
                sbatch_mem[start + i] = src[i]


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
def _histogram_build(keys, p):
    """Build histogram of key values for small p. O(n)."""
    size = np.int64(1) << np.int64(p)
    hist = np.zeros(size, dtype=np.int64)
    for i in range(keys.size):
        hist[keys[i]] += 1
    return hist


@njit(nogil=True, fastmath=True)
def _histogram_get_item(seg_tree, n_leaves, position):
    """Get the key at sorted position via segment tree walk. O(p)."""
    pos = np.int64(position)
    node = np.int64(1)
    while node < n_leaves:
        left = 2 * node
        left_count = seg_tree[left]
        if pos < left_count:
            node = left
        else:
            pos -= left_count
            node = left + 1
    return np.uint32(node - n_leaves)


@njit(nogil=True, fastmath=True)
def _histogram_reconstruct(hist, total):
    """Reconstruct all sorted keys from histogram. O(n + 2^p)."""
    out = np.empty(total, dtype=np.uint32)
    idx = np.int64(0)
    for v in range(hist.size):
        c = hist[v]
        for _ in range(c):
            out[idx] = np.uint32(v)
            idx += 1
    return out


@njit(nogil=True, fastmath=True)
def _count_less_than(sbatch_mem, start, end, val):
    """Binary search: count entries < val in sorted region [start, end)."""
    lo = start
    hi = end
    while lo < hi:
        mid = (lo + hi) >> 1
        if sbatch_mem[mid] < val:
            lo = mid + 1
        else:
            hi = mid
    return lo - start


@njit(nogil=True, fastmath=True)
def _count_less_equal(sbatch_mem, start, end, val):
    """Binary search: count entries <= val in sorted region [start, end)."""
    lo = start
    hi = end
    while lo < hi:
        mid = (lo + hi) >> 1
        if sbatch_mem[mid] <= val:
            lo = mid + 1
        else:
            hi = mid
    return lo - start


@njit(nogil=True, fastmath=True)
def _get_item(sbatch_mem, seg_tree, bin_cumulative, n_bins,
              entry_bits, batch_boundaries, n_batches, position):
    """Get the key at a given sorted position.

    1. Segment tree walk to find bin
    2. K-way selection across sorted runs via binary search
    3. Reconstruct key = (bin_id << entry_bits) | entry
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
    bin_pos = pos  # position within this bin (0-indexed)
    bin_start = bin_cumulative[bin_id]

    # Collect sorted run boundaries (absolute positions in sbatch_mem)
    run_starts = np.empty(n_batches, dtype=np.int64)
    run_ends = np.empty(n_batches, dtype=np.int64)
    n_runs = 0
    for k in range(n_batches):
        s = bin_start + batch_boundaries[bin_id, k]
        e = bin_start + batch_boundaries[bin_id, k + 1]
        if s < e:
            run_starts[n_runs] = s
            run_ends[n_runs] = e
            n_runs += 1

    if n_runs == 1:
        # Single run — direct index
        entry = sbatch_mem[run_starts[0] + bin_pos]
        return (np.uint32(bin_id) << np.uint32(entry_bits)) | entry

    # K-way selection: find the element at rank bin_pos across k sorted runs.
    # Binary search on value: find smallest v such that total elements <= v
    # across all runs is >= bin_pos + 1.
    # First, find min and max across all runs.
    lo_val = np.uint32(0xFFFFFFFF)
    hi_val = np.uint32(0)
    for r in range(n_runs):
        v = sbatch_mem[run_starts[r]]
        if v < lo_val:
            lo_val = v
        v = sbatch_mem[run_ends[r] - 1]
        if v > hi_val:
            hi_val = v

    # Binary search on value
    target = bin_pos + 1  # we want the target-th smallest
    while lo_val < hi_val:
        mid_val = lo_val + ((hi_val - lo_val) >> 1)
        # Count elements <= mid_val across all runs
        total_le = np.int64(0)
        for r in range(n_runs):
            total_le += _count_less_equal(sbatch_mem, run_starts[r],
                                          run_ends[r], mid_val)
        if total_le < target:
            lo_val = mid_val + np.uint32(1)
        else:
            hi_val = mid_val

    # lo_val is the answer entry value
    entry = lo_val
    return (np.uint32(bin_id) << np.uint32(entry_bits)) | entry


@njit(nogil=True, fastmath=True)
def _reconstruct_all(sbatch_mem, bin_counts, bin_cumulative,
                     n_bins, entry_bits, batch_boundaries, n_batches):
    """Reconstruct all keys in sorted order via k-way merge of sorted runs."""
    total = np.int64(0)
    for b in range(n_bins):
        total += np.int64(bin_counts[b])

    keys_out = np.empty(total, dtype=np.uint32)
    out_idx = np.int64(0)

    for b in range(n_bins):
        count = np.int64(bin_counts[b])
        if count == 0:
            continue

        bin_start = bin_cumulative[b]
        key_high = np.uint32(b) << np.uint32(entry_bits)

        # Collect non-empty runs
        heads = np.empty(n_batches, dtype=np.int64)
        ends = np.empty(n_batches, dtype=np.int64)
        n_runs = 0
        for k in range(n_batches):
            s = bin_start + batch_boundaries[b, k]
            e = bin_start + batch_boundaries[b, k + 1]
            if s < e:
                heads[n_runs] = s
                ends[n_runs] = e
                n_runs += 1

        if n_runs == 1:
            # Single run — just copy
            for i in range(heads[0], ends[0]):
                keys_out[out_idx] = key_high | sbatch_mem[i]
                out_idx += 1
        else:
            # K-way merge
            for _ in range(count):
                best = -1
                best_val = np.uint32(0xFFFFFFFF)
                for r in range(n_runs):
                    if heads[r] < ends[r]:
                        v = sbatch_mem[heads[r]]
                        if v < best_val:
                            best = r
                            best_val = v
                keys_out[out_idx] = key_high | best_val
                out_idx += 1
                heads[best] += 1
                # Remove exhausted run
                if heads[best] >= ends[best]:
                    n_runs -= 1
                    if best < n_runs:
                        heads[best] = heads[n_runs]
                        ends[best] = ends[n_runs]

    return keys_out


# === High-level API ===

class HistogramSortResult:
    """Result of histogram sort for small p. No bins, no scatter."""

    def __init__(self, hist, seg_tree, p):
        self.hist = hist
        self.seg_tree = seg_tree
        self.p = p
        self.n_leaves = np.int64(1) << np.int64(p)
        self._n = int(np.sum(hist))

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
        """O(p) tree walk — leaf index is the key value."""
        return _histogram_get_item(self.seg_tree, self.n_leaves, np.int64(position))

    def reconstruct_all(self):
        """O(n + 2^p) histogram expansion."""
        return _histogram_reconstruct(self.hist, np.int64(self._n))


class FractalSortResult:
    """Result of fractal sort. Supports indexed access to sorted keys."""

    def __init__(self, sbatch_mem, bin_counts, bin_cumulative,
                 ln, lb, p, n_bins, seg_tree, batch_boundaries, n_batches):
        self.sbatch_mem = sbatch_mem
        self.bin_counts = bin_counts
        self.bin_cumulative = bin_cumulative
        self.seg_tree = seg_tree
        self.batch_boundaries = batch_boundaries
        self.n_batches = n_batches
        self.ln = ln
        self.lb = lb
        self.p = p
        self.n_bins = n_bins
        self.ln_minus_lb = ln - lb
        self.entry_bits = lb + (p - ln)
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
        """Get the key at a given sorted position. O(log n_bins + k*log(bin_size))."""
        return _get_item(self.sbatch_mem, self.seg_tree,
                         self.bin_cumulative, self.n_bins, self.entry_bits,
                         self.batch_boundaries, self.n_batches,
                         np.int64(position))

    def reconstruct_all(self):
        """Reconstruct all keys in sorted order via k-way merge. Returns uint32 array."""
        return _reconstruct_all(self.sbatch_mem, self.bin_counts,
                                self.bin_cumulative, self.n_bins,
                                self.entry_bits, self.batch_boundaries,
                                self.n_batches)


@njit(nogil=True, fastmath=True)
def _fractalsort_core(keys, entry_bits, n_bins, n_batches):
    """Core sort pipeline: precount + scatter."""
    n = keys.size
    batch_size = (n + n_batches - 1) // n_batches

    # Precount bins
    all_bc = np.zeros(n_bins, dtype=np.int64)
    for i in range(n):
        bid = np.int64(keys[i] >> np.uint32(entry_bits))
        if bid < n_bins:
            all_bc[bid] += 1

    # Bin starts (prefix sum)
    bin_starts = np.zeros(n_bins, dtype=np.int64)
    for i in range(1, n_bins):
        bin_starts[i] = bin_starts[i - 1] + all_bc[i - 1]

    # Scatter entries
    sbatch_mem = np.empty(n, dtype=np.uint32)
    bin_write_pos = bin_starts.copy()
    batch_boundaries = np.zeros((n_bins, n_batches + 1), dtype=np.int64)
    entry_mask = np.uint32((1 << entry_bits) - 1)

    for b_idx in range(n_batches):
        i0 = b_idx * batch_size
        i1 = i0 + batch_size
        if i1 > n:
            i1 = n
        if i0 >= n:
            for k in range(b_idx, n_batches):
                for bi in range(n_bins):
                    batch_boundaries[bi, k + 1] = batch_boundaries[bi, k]
            break
        for bi in range(n_bins):
            batch_boundaries[bi, b_idx] = bin_write_pos[bi] - bin_starts[bi]
        for i in range(i0, i1):
            ki = keys[i]
            bid = np.int64(ki >> np.uint32(entry_bits))
            if bid >= n_bins:
                bid = np.int64(n_bins - 1)
            entry = ki & entry_mask
            wp = bin_write_pos[bid]
            sbatch_mem[wp] = entry
            bin_write_pos[bid] = wp + np.int64(1)
        for bi in range(n_bins):
            batch_boundaries[bi, b_idx + 1] = bin_write_pos[bi] - bin_starts[bi]

    bin_counts = np.empty(n_bins, dtype=np.int64)
    bin_cumulative = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        bin_counts[i] = bin_write_pos[i] - bin_starts[i]
    for i in range(1, n_bins):
        bin_cumulative[i] = bin_cumulative[i - 1] + bin_counts[i - 1]

    return sbatch_mem, bin_counts, bin_cumulative, batch_boundaries



def fractalsort(keys, p=32, lb=None, n_batches=4):
    """Sort keys using fractal sort.

    Args:
        keys: uint32 array of p-bit keys
        p: key precision in bits (default 32)
        lb: log2(bin size). Controls bins = 2^(ln-lb). Default: ln - 8.
        n_batches: number of processing batches

    Returns:
        FractalSortResult with indexed access to sorted keys.
    """
    n = keys.size
    e = int(np.ceil(np.log2(n))) if n > 1 else 1
    ln = e

    # Small p: histogram fits in cache, no bins needed
    if p <= 20 and e >= p:
        hist = _histogram_build(keys, p)
        n_leaves = 1 << p
        seg_tree = _build_seg_tree(hist, n_leaves)
        return HistogramSortResult(hist, seg_tree, p)

    if lb is None:
        lb = e - 10 if e <= 20 else e - 6
    lb = max(0, min(lb, ln - 1))
    ln_minus_lb = ln - lb
    n_bins = 1 << ln_minus_lb
    entry_bits = lb + (p - ln)

    sbatch_mem, bin_counts, bin_cumulative, batch_boundaries = \
        _fractalsort_core(keys, entry_bits, n_bins, n_batches)

    # Sort entries within each bin — only ceil(entry_bits/8) radix passes
    max_passes = (entry_bits + 7) // 8
    for b_idx in range(n_batches):
        starts_abs = bin_cumulative + batch_boundaries[:, b_idx]
        ends_abs = bin_cumulative + batch_boundaries[:, b_idx + 1]
        _sort_batch_in_bins(sbatch_mem, starts_abs, ends_abs, n_bins,
                            max_passes)

    # Segment tree for get_item
    seg_tree = _build_seg_tree(bin_counts, n_bins)

    return FractalSortResult(
        sbatch_mem=sbatch_mem,
        bin_counts=bin_counts,
        bin_cumulative=bin_cumulative,
        ln=ln, lb=lb, p=p, n_bins=n_bins,
        seg_tree=seg_tree,
        batch_boundaries=batch_boundaries,
        n_batches=n_batches,
    )
