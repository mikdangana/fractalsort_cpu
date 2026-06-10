"""Run a single e-value benchmark. Usage: python bench_wsl_single.py <e>"""
import sys, os, numpy as np, time

from fractalsort_core.frmw_io_fast import (frmw_io_fast_process, build_counters_from_hist,
                                           _build_reverse_lut, _bit_reverse_lut32)
from numba import njit

p = 32; cache_mb = 8; wc_margin = 2; n_batches = 4
# Keep n_io_bins = 2^10 = 1024 for all e by setting lb = e - 10
def get_lb(e):
    return max(10, e - 10)

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
def precount_bins_fast(keys, ln_minus_lb, lut):
    n_bins = np.int64(1) << ln_minus_lb
    mask = np.int64(n_bins - 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(keys.size):
        bid = _bit_reverse_lut32(np.int64(keys[i]) & mask, ln_minus_lb, lut)
        counts[bid] += 1
    return counts

def build_layout(e, lb):
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

def bench_one(e, n_runs=5):
    n = 1 << e
    lb = get_lb(e)
    lut = _build_reverse_lut()

    # Warmup
    k_warm = np.random.randint(0, 2**32, size=1 << 14, dtype=np.uint32)
    _kdb_radix_sort_uint32(k_warm)

    ln = e; ln_minus_lb = ln - lb
    n_io_bins = 1 << ln_minus_lb
    batch_size = n // n_batches
    lc, widths_c, per_word_c, word_offsets_c, total_words = build_layout(e, lb)

    np.random.seed(42)
    keys = np.random.randint(0, 1 << p, size=n, dtype=np.uint32)

    print(f"e={e} n={n:,} lb={lb} n_io_bins={n_io_bins}")

    # FS warmup
    all_bc = precount_bins_fast(keys, ln_minus_lb, lut)
    bin_starts = np.zeros(n_io_bins, dtype=np.int64)
    for i in range(1, n_io_bins):
        bin_starts[i] = bin_starts[i - 1] + all_bc[i - 1]

    for _ in range(2):
        hist = np.zeros(n_io_bins, dtype=np.int64)
        sbatch_mem = np.empty(n, dtype=np.uint32)
        bin_counts = np.zeros(n_io_bins, dtype=np.int32)
        bin_wp = bin_starts.copy()
        for b in range(n_batches):
            i0, i1 = b * batch_size, (b + 1) * batch_size
            frmw_io_fast_process(keys[i0:i1], hist, ln_minus_lb, lb, p, ln,
                                 sbatch_mem, bin_wp, bin_counts, lut)
        cnts = np.zeros(total_words, dtype=np.uint64)
        build_counters_from_hist(hist, cnts, widths_c, per_word_c, word_offsets_c, lc, n_io_bins)

    # kdb+ warmup
    for _ in range(2):
        _kdb_radix_sort_uint32(keys)

    # Timed: kdb+
    kdb_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _kdb_radix_sort_uint32(keys)
        kdb_times.append(time.perf_counter() - t0)

    # Timed: FS
    fs_times = []
    for _ in range(n_runs):
        hist = np.zeros(n_io_bins, dtype=np.int64)
        sbatch_mem = np.empty(n, dtype=np.uint32)
        bin_counts = np.zeros(n_io_bins, dtype=np.int32)
        bin_wp = bin_starts.copy()
        t0 = time.perf_counter()
        for b in range(n_batches):
            i0, i1 = b * batch_size, (b + 1) * batch_size
            frmw_io_fast_process(keys[i0:i1], hist, ln_minus_lb, lb, p, ln,
                                 sbatch_mem, bin_wp, bin_counts, lut)
        cnts = np.zeros(total_words, dtype=np.uint64)
        build_counters_from_hist(hist, cnts, widths_c, per_word_c, word_offsets_c, lc, n_io_bins)
        fs_times.append(time.perf_counter() - t0)

    t_kdb = min(kdb_times)
    t_fs = min(fs_times)
    kdb_mks = n / t_kdb / 1e6
    fs_mks = n / t_fs / 1e6
    speedup = fs_mks / kdb_mks
    win = '<-- FS wins' if speedup > 1.0 else ''
    print(f'e={e} n={n:>12,}  kdb+={kdb_mks:>7.1f}M/s  FS={fs_mks:>7.1f}M/s  speedup={speedup:.2f}x  {win}')

if __name__ == '__main__':
    e = int(sys.argv[1])
    bench_one(e)
