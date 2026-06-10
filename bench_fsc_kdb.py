"""Benchmark FSF vs FSC8 vs kdb+ radix sort.

FSF:   scatter + packed histogram bin sort (single pass, 4-bit counters)
FSFA:  FSF + reconstruct full 32-bit keys
FSC8:  scatter + 8-bit radix bin sort (2 passes for 16-bit entries)
FSCA8: FSC8 + reconstruct full 32-bit keys
kdb+:  LSB radix sort (256-way, 4-pass, 32-bit)

All keys are uint32, p=32, lc=16 (65536 bins, 16-bit entries).
"""
import numpy as np
import time
import sys
import gc

from fractalsort_core.frmw_io_fast import (
    frmw_io_fast_process, countsort_bins_8bit,
    reconstruct_keys, _build_reverse_lut,
)
from numba import njit

p = 32
n_runs = 1


@njit(nogil=True, fastmath=True)
def _kdb_radix_sort_uint32(keys):
    n = keys.size
    src = keys.copy()
    buf = np.empty(n, dtype=np.uint32)
    for pass_idx in range(4):
        shift = np.uint32(pass_idx * 8)
        counts = np.zeros(256, dtype=np.int64)
        for i in range(n):
            b = (src[i] >> shift) & np.uint32(0xFF)
            counts[b] += 1
        offsets = np.zeros(256, dtype=np.int64)
        for b in range(1, 256):
            offsets[b] = offsets[b - 1] + counts[b - 1]
        for i in range(n):
            b = (src[i] >> shift) & np.uint32(0xFF)
            buf[offsets[b]] = src[i]
            offsets[b] += 1
        src, buf = buf, src
    return src


@njit(nogil=True, fastmath=True)
def _precount_and_starts(keys, bin_shift, n_bins):
    all_bc = np.zeros(n_bins, dtype=np.int64)
    for i in range(keys.size):
        bid = np.int64(keys[i] >> bin_shift)
        all_bc[bid] += 1
    bin_starts = np.zeros(n_bins, dtype=np.int64)
    for i in range(1, n_bins):
        bin_starts[i] = bin_starts[i - 1] + all_bc[i - 1]
    return all_bc, bin_starts


def setup_frmw(keys, e, lb, lut):
    ln = e
    ln_minus_lb = ln - lb
    n_bins = 1 << ln_minus_lb
    n = keys.size
    bin_shift = np.uint32(p - ln_minus_lb)

    all_bc, bin_starts = _precount_and_starts(keys, bin_shift, n_bins)

    hist = np.zeros(n_bins, dtype=np.int64)
    sbatch_mem = np.empty(n, dtype=np.uint32)
    bin_counts = np.zeros(n_bins, dtype=np.int32)
    bin_wp = bin_starts.copy()

    frmw_io_fast_process(keys, hist, ln_minus_lb, lb, p, ln,
                         sbatch_mem, bin_starts, bin_counts, lut, bin_wp)

    return sbatch_mem, bin_starts, bin_counts, hist, n_bins, ln_minus_lb


# === FSF: packed histogram bin sort ===

@njit(nogil=True, fastmath=True)
def fsf_binsort(sbatch_mem, bin_starts, bin_counts, n_bins, entry_bits):
    """Sort entries within each bin using a direct packed histogram.

    Single pass to build histogram of entry values, then reconstruct
    sorted entries by walking the histogram.
    entry_bits must be <= 16 for the histogram to fit in L1.
    """
    n_buckets = np.int64(1) << entry_bits
    # 4-bit packed counters: 2 counters per byte, n_buckets/2 bytes
    # Use uint8 array where each byte holds 2 x 4-bit counters
    packed_size = (n_buckets + 1) // 2
    packed_hist = np.zeros(packed_size, dtype=np.uint8)

    for bid in range(n_bins):
        c = np.int64(bin_counts[bid])
        if c == 0:
            continue

        start = bin_starts[bid]

        # Clear histogram (only need to clear what we'll use)
        for j in range(packed_size):
            packed_hist[j] = 0

        # Single pass: count entries
        for ki in range(c):
            val = np.int64(sbatch_mem[start + ki])
            byte_idx = val >> 1
            if val & 1:
                packed_hist[byte_idx] += np.uint8(0x10)  # high nibble
            else:
                packed_hist[byte_idx] += np.uint8(0x01)  # low nibble

        # Reconstruct sorted entries by walking histogram
        out_pos = start
        for val in range(n_buckets):
            byte_idx = val >> 1
            if val & 1:
                cnt = (packed_hist[byte_idx] >> 4) & 0x0F
            else:
                cnt = packed_hist[byte_idx] & 0x0F
            for _ in range(cnt):
                sbatch_mem[out_pos] = np.uint32(val)
                out_pos += 1


@njit(nogil=True, fastmath=True)
def fsf_binsort_full(sbatch_mem, bin_starts, bin_counts, n_bins, entry_bits):
    """Sort entries within each bin using a direct histogram.

    Uses full int32 counters (no packing) to avoid 4-bit overflow.
    entry_bits must be <= 16 for the histogram to fit in L1/L2.
    """
    n_buckets = np.int64(1) << entry_bits
    hist = np.zeros(n_buckets, dtype=np.int32)

    for bid in range(n_bins):
        c = np.int64(bin_counts[bid])
        if c == 0:
            continue

        start = bin_starts[bid]

        # Clear
        for j in range(n_buckets):
            hist[j] = 0

        # Single pass: count entries
        for ki in range(c):
            hist[np.int64(sbatch_mem[start + ki])] += 1

        # Reconstruct sorted entries
        out_pos = start
        for val in range(n_buckets):
            cnt = hist[val]
            for _ in range(cnt):
                sbatch_mem[out_pos] = np.uint32(val)
                out_pos += 1


@njit(nogil=True, fastmath=True)
def fsf_binsort_2bit(sbatch_mem, bin_starts, bin_counts, n_bins, entry_bits):
    """Sort entries using 2-bit packed histogram.

    4 counters per byte. Max count per bucket = 3.
    For 16-bit entries: 2^16 / 4 = 16KB — fits L1.
    Overflow (count > 3) wraps silently — only valid for small bins.
    """
    n_buckets = np.int64(1) << entry_bits
    packed_size = (n_buckets + 3) // 4  # 4 counters per byte
    packed_hist = np.zeros(packed_size, dtype=np.uint8)

    for bid in range(n_bins):
        c = np.int64(bin_counts[bid])
        if c == 0:
            continue

        start = bin_starts[bid]

        for j in range(packed_size):
            packed_hist[j] = 0

        # Single pass: count entries (2-bit per counter)
        for ki in range(c):
            val = np.int64(sbatch_mem[start + ki])
            byte_idx = val >> 2           # 4 counters per byte
            shift = (val & 3) << 1        # 0, 2, 4, or 6
            packed_hist[byte_idx] += np.uint8(1 << shift)

        # Reconstruct sorted entries
        out_pos = start
        for val in range(n_buckets):
            byte_idx = val >> 2
            shift = (val & 3) << 1
            cnt = (packed_hist[byte_idx] >> shift) & 0x03
            for _ in range(cnt):
                sbatch_mem[out_pos] = np.uint32(val)
                out_pos += 1


@njit(nogil=True, fastmath=True)
def fsf_binsort_2bit_sparse(sbatch_mem, bin_starts, bin_counts, n_bins, entry_bits):
    """Sort entries using 2-bit packed histogram with sparse tracking.

    Only touches buckets that have entries. Clears only dirty locations.
    Work is O(bin_size) not O(2^entry_bits).
    """
    n_buckets = np.int64(1) << entry_bits
    packed_size = (n_buckets + 3) // 4
    packed_hist = np.zeros(packed_size, dtype=np.uint8)

    # Dirty tracking: worst case each entry hits a unique bucket
    max_bin = np.int64(0)
    for bid in range(n_bins):
        if np.int64(bin_counts[bid]) > max_bin:
            max_bin = np.int64(bin_counts[bid])
    dirty_bytes = np.empty(max_bin, dtype=np.int64)
    dirty_vals = np.empty(max_bin, dtype=np.int64)

    for bid in range(n_bins):
        c = np.int64(bin_counts[bid])
        if c == 0:
            continue

        start = bin_starts[bid]
        n_dirty_bytes = 0
        n_dirty_vals = 0

        # Single pass: count entries, track dirty byte positions
        for ki in range(c):
            val = np.int64(sbatch_mem[start + ki])
            byte_idx = val >> 2
            shift = (val & 3) << 1
            old = packed_hist[byte_idx]
            if old == 0:
                dirty_bytes[n_dirty_bytes] = byte_idx
                n_dirty_bytes += 1
            packed_hist[byte_idx] = old + np.uint8(1 << shift)
            # Track unique values for sorted reconstruction
            # (check if this val's counter went from 0 to 1)
            cnt_before = (old >> shift) & 0x03
            if cnt_before == 0:
                dirty_vals[n_dirty_vals] = val
                n_dirty_vals += 1

        # Sort dirty values (insertion sort — small list)
        for i in range(1, n_dirty_vals):
            v = dirty_vals[i]
            j = i - 1
            while j >= 0 and dirty_vals[j] > v:
                dirty_vals[j + 1] = dirty_vals[j]
                j -= 1
            dirty_vals[j + 1] = v

        # Reconstruct sorted entries from dirty vals only
        out_pos = start
        for di in range(n_dirty_vals):
            val = dirty_vals[di]
            byte_idx = val >> 2
            shift = (val & 3) << 1
            cnt = (packed_hist[byte_idx] >> shift) & 0x03
            for _ in range(cnt):
                sbatch_mem[out_pos] = np.uint32(val)
                out_pos += 1

        # Clear only dirty bytes
        for di in range(n_dirty_bytes):
            packed_hist[dirty_bytes[di]] = 0


@njit(nogil=True, fastmath=True)
def fsf_binsort_4bit_sparse(sbatch_mem, bin_starts, bin_counts, n_bins, entry_bits):
    """FSF default: 4-bit packed histogram, sparse dirty tracking.

    2 counters per byte (high/low nibble). Max count per bucket = 15.
    For 16-bit entries: 2^16 / 2 = 32KB — fits L1.
    Work is O(bin_size) per bin via sparse tracking.
    """
    n_buckets = np.int64(1) << entry_bits
    packed_size = (n_buckets + 1) // 2
    packed_hist = np.zeros(packed_size, dtype=np.uint8)

    max_bin = np.int64(0)
    for bid in range(n_bins):
        if np.int64(bin_counts[bid]) > max_bin:
            max_bin = np.int64(bin_counts[bid])
    dirty_bytes = np.empty(max_bin, dtype=np.int64)
    dirty_vals = np.empty(max_bin, dtype=np.int64)

    for bid in range(n_bins):
        c = np.int64(bin_counts[bid])
        if c == 0:
            continue

        start = bin_starts[bid]
        n_dirty_bytes = 0
        n_dirty_vals = 0

        # Single pass: count entries
        for ki in range(c):
            val = np.int64(sbatch_mem[start + ki])
            byte_idx = val >> 1
            old = packed_hist[byte_idx]
            if old == 0:
                dirty_bytes[n_dirty_bytes] = byte_idx
                n_dirty_bytes += 1
            if val & 1:
                cnt_before = (old >> 4) & 0x0F
                packed_hist[byte_idx] = old + np.uint8(0x10)
            else:
                cnt_before = old & 0x0F
                packed_hist[byte_idx] = old + np.uint8(0x01)
            if cnt_before == 0:
                dirty_vals[n_dirty_vals] = val
                n_dirty_vals += 1

        # Sort dirty values (insertion sort)
        for i in range(1, n_dirty_vals):
            v = dirty_vals[i]
            j = i - 1
            while j >= 0 and dirty_vals[j] > v:
                dirty_vals[j + 1] = dirty_vals[j]
                j -= 1
            dirty_vals[j + 1] = v

        # Reconstruct from dirty vals only
        out_pos = start
        for di in range(n_dirty_vals):
            val = dirty_vals[di]
            byte_idx = val >> 1
            if val & 1:
                cnt = (packed_hist[byte_idx] >> 4) & 0x0F
            else:
                cnt = packed_hist[byte_idx] & 0x0F
            for _ in range(cnt):
                sbatch_mem[out_pos] = np.uint32(val)
                out_pos += 1

        # Clear only dirty bytes
        for di in range(n_dirty_bytes):
            packed_hist[dirty_bytes[di]] = 0


@njit(nogil=True, fastmath=True)
def fsf_binsort_2bit_fresh(sbatch_mem, bin_starts, bin_counts, n_bins, entry_bits):
    """Sort entries using 2-bit packed histogram, fresh alloc each bin.

    Allocates a new zeroed histogram per bin (no clear loop needed).
    Tracks dirty vals for sparse reconstruction.
    """
    n_buckets = np.int64(1) << entry_bits
    packed_size = (n_buckets + 3) // 4

    max_bin = np.int64(0)
    for bid in range(n_bins):
        if np.int64(bin_counts[bid]) > max_bin:
            max_bin = np.int64(bin_counts[bid])
    dirty_vals = np.empty(max_bin, dtype=np.int64)

    for bid in range(n_bins):
        c = np.int64(bin_counts[bid])
        if c == 0:
            continue

        start = bin_starts[bid]
        packed_hist = np.zeros(packed_size, dtype=np.uint8)
        n_dirty_vals = 0

        for ki in range(c):
            val = np.int64(sbatch_mem[start + ki])
            byte_idx = val >> 2
            shift = (val & 3) << 1
            old = packed_hist[byte_idx]
            cnt_before = (old >> shift) & 0x03
            if cnt_before == 0:
                dirty_vals[n_dirty_vals] = val
                n_dirty_vals += 1
            packed_hist[byte_idx] = old + np.uint8(1 << shift)

        # Sort dirty values (insertion sort)
        for i in range(1, n_dirty_vals):
            v = dirty_vals[i]
            j = i - 1
            while j >= 0 and dirty_vals[j] > v:
                dirty_vals[j + 1] = dirty_vals[j]
                j -= 1
            dirty_vals[j + 1] = v

        # Reconstruct from dirty vals only
        out_pos = start
        for di in range(n_dirty_vals):
            val = dirty_vals[di]
            byte_idx = val >> 2
            shift = (val & 3) << 1
            cnt = (packed_hist[byte_idx] >> shift) & 0x03
            for _ in range(cnt):
                sbatch_mem[out_pos] = np.uint32(val)
                out_pos += 1


def bench_fn(fn, n_runs=7, warmup=2):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        times.append(dt)
    times.sort()
    return times[0], times[len(times) // 2]


def main():
    lut = _build_reverse_lut()

    # Warmup JIT
    print("JIT warmup...")
    k_warm = np.random.randint(0, 2**32, size=1 << 14, dtype=np.uint32)
    _kdb_radix_sort_uint32(k_warm)
    sm, bs, bc, h, nb, lnlb = setup_frmw(k_warm, 14, 4, lut)
    countsort_bins_8bit(sm.copy(), bs.copy(), bc.copy(), nb, 4, lnlb, p, 14)
    fsf_binsort(sm.copy(), bs.copy(), bc.copy(), nb, 4 + (p - 14))
    fsf_binsort_full(sm.copy(), bs.copy(), bc.copy(), nb, 4 + (p - 14))
    fsf_binsort_2bit(sm.copy(), bs.copy(), bc.copy(), nb, 4 + (p - 14))
    fsf_binsort_2bit_sparse(sm.copy(), bs.copy(), bc.copy(), nb, 4 + (p - 14))
    fsf_binsort_2bit_fresh(sm.copy(), bs.copy(), bc.copy(), nb, 4 + (p - 14))
    fsf_binsort_4bit_sparse(sm.copy(), bs.copy(), bc.copy(), nb, 4 + (p - 14))
    out = np.empty(k_warm.size, dtype=np.uint32)
    reconstruct_keys(sm, bs, bc, nb, lnlb, p, out)
    print("Done.\n")

    lc = 16
    print("=" * 110)
    print(f"FSF vs FSC8 vs kdb+ radix  (p={p}, lc={lc}, entry_bits={p-lc}, uint32)")
    print("=" * 110)
    print()

    print(f"{'e':>4} {'n':>12} {'lb':>3} {'bins':>6} {'entry':>5} {'bin_sz':>6} | "
          f"{'kdb+':>10} {'FSC8':>10} | {'ratio':>6}")
    print("-" * 70)

    for e in [22, 24, 28]:
        n = 1 << e
        ln = e
        lb = e - 10
        ln_minus_lb = ln - lb
        n_bins = 1 << ln_minus_lb
        entry_bits = lb + (p - ln)
        avg_bin = n // n_bins

        np.random.seed(42)
        keys = np.random.randint(0, 1 << p, size=n, dtype=np.uint32)

        # kdb+ radix
        t_kdb, _ = bench_fn(lambda k=keys: _kdb_radix_sort_uint32(k), n_runs)

        # Setup scatter
        sm, bs, bc, hist, nb, lnlb = setup_frmw(keys, e, lb, lut)

        # FSC8: 8-bit radix bin sort
        def run_fsc8(sm=sm, bs=bs, bc=bc):
            s = sm.copy()
            countsort_bins_8bit(s, bs.copy(), bc.copy(), nb, lb, lnlb, p, ln)
        t_fsc8, _ = bench_fn(run_fsc8, n_runs)

        kdb_mks = n / t_kdb / 1e6
        fsc8_mks = n / t_fsc8 / 1e6

        print(f"{e:>4} {n:>12,} {lb:>3} {n_bins:>6} {entry_bits:>5} {avg_bin:>6} | "
              f"{kdb_mks:>8.1f}M/s {fsc8_mks:>8.1f}M/s | {fsc8_mks/kdb_mks:>5.2f}x")

    print()
    print("All = bin sort only (scatter excluded)")
    print("FSF = 4-bit packed histogram, sparse dirty tracking (32KB for 16-bit entries)")
    print()
    print("Environment:")
    print(f"  Platform:    {sys.platform}")
    print(f"  Python:      {sys.version.split()[0]}")
    import numba
    print(f"  Numba:       {numba.__version__}")
    print(f"  NumPy:       {np.__version__}")
    print(f"  p={p}, lc={lc}, n_runs={n_runs}")


if __name__ == "__main__":
    main()
