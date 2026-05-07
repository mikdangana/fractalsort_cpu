"""Benchmark frmw_io_fast process phase vs radix sort."""
import numpy as np
import time
import sys

sys.path.insert(0, '.')
from frmw_io_fast import (frmw_io_fast_process, build_counters_from_hist,
                           _build_reverse_lut, _bit_reverse_lut32)
from numba import njit

p = 32; lb = 10; cache_mb = 8; wc_margin = 2; n_batches = 4
n_runs = 3


@njit(nogil=True, fastmath=True)
def precount_bins_fast(keys, ln_minus_lb, lut):
    n_bins = np.int64(1) << ln_minus_lb
    mask = np.int64(n_bins - 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(keys.size):
        bid = _bit_reverse_lut32(np.int64(keys[i]) & mask, ln_minus_lb, lut)
        counts[bid] += 1
    return counts


@njit(nogil=True, fastmath=True)
def radix_uint32(keys):
    n = keys.size
    output = np.empty(n, dtype=np.uint32)
    for shift in range(0, 32, 8):
        counts = np.zeros(256, dtype=np.int64)
        for i in range(n):
            counts[(keys[i] >> shift) & 0xFF] += 1
        offsets = np.zeros(257, dtype=np.int64)
        for i in range(256):
            offsets[i + 1] = offsets[i] + counts[i]
        for i in range(n):
            b = (keys[i] >> shift) & 0xFF
            output[offsets[b]] = keys[i]
            offsets[b] += 1
        keys = output.copy()
    return output


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


# Build LUT
print('Building LUT + compiling...')
lut = _build_reverse_lut()

# Warmup
e0 = 14
lc0, w0, pw0, wo0, tw0 = build_layout(e0)
k0 = np.random.randint(0, 1 << p, size=1 << e0, dtype=np.uint32)
bc0 = precount_bins_fast(k0, e0 - lb, lut)
hist0 = np.zeros(1 << (e0 - lb), dtype=np.int64)
sb0 = np.empty(1 << e0, dtype=np.uint32)
sbc0 = np.zeros(1 << (e0 - lb), dtype=np.int32)
starts0 = np.zeros(1 << (e0 - lb), dtype=np.int64)
for i in range(1, 1 << (e0 - lb)):
    starts0[i] = starts0[i - 1] + bc0[i - 1]
wp0 = starts0.copy()
frmw_io_fast_process(k0, hist0, e0 - lb, lb, p, e0, sb0, wp0, sbc0, lut)
cnts0 = np.zeros(tw0, dtype=np.uint64)
build_counters_from_hist(hist0, cnts0, w0, pw0, wo0, lc0, 1 << (e0 - lb))
_ = radix_uint32(k0)
print('Done.\n')

print(f"{'e':>4} {'n':>12} {'lc':>3} {'bins':>8} | {'io_proc':>10} {'radix':>10} | {'io/rx':>6}")
print('-' * 70)

for e in [14, 16, 18, 20, 22, 24]:
    n = 1 << e
    ln = e
    batch_size = n // n_batches
    ln_minus_lb = ln - lb
    n_io_bins = 1 << ln_minus_lb
    entry_bits = lb + (p - ln)
    lc, widths_c, per_word_c, word_offsets_c, total_words = build_layout(e)

    np.random.seed(42)
    keys = np.random.randint(0, 1 << p, size=n, dtype=np.uint32)
    all_bc = precount_bins_fast(keys, ln_minus_lb, lut)
    bin_starts = np.zeros(n_io_bins, dtype=np.int64)
    for i in range(1, n_io_bins):
        bin_starts[i] = bin_starts[i - 1] + all_bc[i - 1]

    # Timing: frmw_io process + counter build
    proc_best = float('inf')
    for run in range(n_runs):
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
        t1 = time.perf_counter()
        proc_best = min(proc_best, t1 - t0)

    # Radix
    radix_best = float('inf')
    for run in range(n_runs):
        t0 = time.perf_counter()
        _ = radix_uint32(keys)
        radix_best = min(radix_best, time.perf_counter() - t0)

    io_mks = n / proc_best / 1e6
    rx_mks = n / radix_best / 1e6
    ratio = proc_best / radix_best

    print(f'{e:>4} {n:>12,} {lc:>3} {n_io_bins:>8,} | '
          f'{io_mks:>8.1f}M/s {rx_mks:>8.1f}M/s | {ratio:>5.2f}x')
