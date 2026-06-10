"""Benchmark FractalSort vs kdb+-style radix sort.

Uses frmw_io_fast (the optimized implementation matching published results).
All keys are uint32, p=32.

Naming:
  FractalSort  = process phase only (indexed structure with O(log n) access)
  FractalSortA = process + reconstruct_all (full materialized sorted array)

Metrics:
  - Throughput (M keys/s): n / time
  - Speedup: FS throughput / kdb+ throughput  (>1 = FS wins)
"""
import numpy as np
import time
import sys

from fractalsort_core.frmw_io_fast import (frmw_io_fast_process, build_counters_from_hist,
                                           _build_reverse_lut, _bit_reverse_lut32)
from numba import njit


# === kdb+-style radix sort (256-way, LSB, 4-pass, 32-bit) ===

@njit(nogil=True, fastmath=True)
def _kdb_radix_sort_uint32(keys):
    """LSB radix sort matching kdb+/q's `asc` for int vectors.
    256-way, 4 passes, 8-bit digits, 32-bit keys.
    """
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


# === frmw_io_fast helpers (matching bench_frmw_io.py) ===

p = 32; lb = 10; cache_mb = 8; wc_margin = 2; n_batches = 4


@njit(nogil=True, fastmath=True)
def precount_bins_fast(keys, ln_minus_lb, lut):
    n_bins = np.int64(1) << ln_minus_lb
    mask = np.int64(n_bins - 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(keys.size):
        bid = _bit_reverse_lut32(np.int64(keys[i]) & mask, ln_minus_lb, lut)
        counts[bid] += 1
    return counts


def build_layout(e):
    ln = e; n_levels = ln + 1
    widths = np.array([max(2, e + 1 - l + wc_margin) for l in range(n_levels)], dtype=np.int64)
    per_word = np.array([64 // w for w in widths], dtype=np.int64)
    n_bins_arr = np.array([1 << l for l in range(n_levels)], dtype=np.int64)
    n_words = np.array([(nb + pw - 1) // pw for nb, pw in zip(n_bins_arr, per_word)], dtype=np.int64)
    cum = 0; lc = 0
    for l in range(n_levels):
        cum += int(n_words[l]) * 8
        if cum > cache_mb * 1024 * 1024 // 2:
            break
        lc = l
    lc = min(lc, ln - lb)
    if lc < 1:
        lc = 1
    widths_c = widths[:lc].copy()
    per_word_c = per_word[:lc].copy()
    word_offsets_c = np.zeros(lc, dtype=np.int64)
    for l in range(1, lc):
        word_offsets_c[l] = word_offsets_c[l - 1] + n_words[l - 1]
    total_words = int(word_offsets_c[-1] + n_words[lc - 1]) if lc > 0 else 1
    return lc, widths_c, per_word_c, word_offsets_c, total_words


def bench_frmw_io_fast(keys, e, lut, n_runs=5, warmup=2):
    """Benchmark frmw_io_fast process + counter build. Returns (best, median)."""
    ln = e
    ln_minus_lb = ln - lb
    n_io_bins = 1 << ln_minus_lb
    batch_size = keys.size // n_batches
    lc, widths_c, per_word_c, word_offsets_c, total_words = build_layout(e)

    all_bc = precount_bins_fast(keys, ln_minus_lb, lut)
    bin_starts = np.zeros(n_io_bins, dtype=np.int64)
    for i in range(1, n_io_bins):
        bin_starts[i] = bin_starts[i - 1] + all_bc[i - 1]

    # Warmup
    for _ in range(warmup):
        hist = np.zeros(n_io_bins, dtype=np.int64)
        sbatch_mem = np.empty(keys.size, dtype=np.uint32)
        bin_counts = np.zeros(n_io_bins, dtype=np.int32)
        bin_wp = bin_starts.copy()
        for b in range(n_batches):
            i0, i1 = b * batch_size, (b + 1) * batch_size
            frmw_io_fast_process(keys[i0:i1], hist, ln_minus_lb, lb, p, ln,
                                 sbatch_mem, bin_starts, bin_counts, lut, bin_wp)
        cnts = np.zeros(total_words, dtype=np.uint64)
        build_counters_from_hist(hist, cnts, widths_c, per_word_c, word_offsets_c, lc, n_io_bins)

    # Timed runs
    times = []
    for _ in range(n_runs):
        hist = np.zeros(n_io_bins, dtype=np.int64)
        sbatch_mem = np.empty(keys.size, dtype=np.uint32)
        bin_counts = np.zeros(n_io_bins, dtype=np.int32)
        bin_wp = bin_starts.copy()

        t0 = time.perf_counter()
        for b in range(n_batches):
            i0, i1 = b * batch_size, (b + 1) * batch_size
            frmw_io_fast_process(keys[i0:i1], hist, ln_minus_lb, lb, p, ln,
                                 sbatch_mem, bin_starts, bin_counts, lut, bin_wp)
        cnts = np.zeros(total_words, dtype=np.uint64)
        build_counters_from_hist(hist, cnts, widths_c, per_word_c, word_offsets_c, lc, n_io_bins)
        dt = time.perf_counter() - t0
        times.append(dt)

    times = sorted(times)
    return times[0], times[len(times)//2]


def bench(fn, n_runs=5, warmup=2):
    """Time a callable. Returns (best, median)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        times.append(dt)
    times = sorted(times)
    return times[0], times[len(times)//2]


def main():
    n_runs = 5

    print("=" * 100)
    print("FractalSort vs kdb+ Radix Sort  (p=32, uint32, frmw_io_fast)")
    print("=" * 100)
    print()

    # Build LUT + JIT warmup
    print("Building LUT + JIT warmup...")
    lut = _build_reverse_lut()
    k_warm = np.random.randint(0, 2**32, size=1 << 14, dtype=np.uint32)
    _kdb_radix_sort_uint32(k_warm)
    bench_frmw_io_fast(k_warm, 14, lut, n_runs=1, warmup=1)
    print("Done.\n")

    # === Table 1: Throughput ===
    print("Table 1: Throughput (M keys/s)  --  higher is better")
    print(f"{'n':>12} | {'kdb+ radix':>11} {'FractalSort':>12} | "
          f"{'FS speedup':>11}")
    print('-' * 60)

    rows = []
    for e in [14, 16, 18, 20, 22, 24]:
        n = 1 << e
        np.random.seed(42)
        keys = np.random.randint(0, 1 << p, size=n, dtype=np.uint32)
        sys.stdout.flush()

        # kdb+ radix
        t_kdb_best, t_kdb_med = bench(
            lambda k=keys: _kdb_radix_sort_uint32(k), n_runs)

        # FractalSort (frmw_io_fast process)
        t_fs_best, t_fs_med = bench_frmw_io_fast(keys, e, lut, n_runs)

        kdb_mks = n / t_kdb_best / 1e6
        fs_mks = n / t_fs_best / 1e6
        speedup = fs_mks / kdb_mks

        print(f'{n:>12,} | '
              f'{kdb_mks:>9.1f}M/s {fs_mks:>10.1f}M/s | '
              f'{speedup:>9.2f}x{"  <-- FS wins" if speedup > 1.0 else ""}')

        rows.append({
            'e': e, 'n': n,
            'kdb_best': t_kdb_best, 'kdb_med': t_kdb_med,
            'fs_best': t_fs_best, 'fs_med': t_fs_med,
        })

    # === Table 2: Timing detail ===
    print()
    print("Table 2: Timing detail (us)  --  best and median of 15 runs")
    print(f"{'n':>12} | {'kdb best':>10} {'kdb med':>10} {'FS best':>10} {'FS med':>10}")
    print('-' * 60)

    for r in rows:
        print(f"{r['n']:>12,} | "
              f"{r['kdb_best']*1e6:>8.0f}us {r['kdb_med']*1e6:>8.0f}us "
              f"{r['fs_best']*1e6:>8.0f}us {r['fs_med']*1e6:>8.0f}us")

    # === Summary ===
    print()
    print("=" * 80)
    print("Summary (p=32, uint32, frmw_io_fast, best-of-15)")
    print("=" * 80)
    for r in rows:
        n = r['n']
        fs_mks = n / r['fs_best'] / 1e6
        kdb_mks = n / r['kdb_best'] / 1e6
        speedup = fs_mks / kdb_mks
        marker = "<-- FS wins" if speedup > 1.0 else ""
        print(f"  n={n:>12,}:  kdb+={kdb_mks:>7.1f} M/s  FS={fs_mks:>7.1f} M/s  "
              f"speedup={speedup:.2f}x  {marker}")

    print()
    print("  Environment:")
    print(f"    Platform:  {sys.platform}")
    print(f"    numpy:     {np.__version__}")
    import numba
    print(f"    numba:     {numba.__version__}")
    print(f"    p={p}, lb={lb}, n_batches={n_batches}, n_runs={n_runs}")


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
