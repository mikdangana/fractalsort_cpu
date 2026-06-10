"""Benchmark FS and kdb+ in separate processes.
Workaround: numba heap corruption after 2+ full pipeline runs at e>=22.
Each subprocess does 1 warmup + 1 timed run. We spawn multiple subprocesses
to get multiple samples.

Usage: python bench_split.py <e> [n_samples] [lb]
"""
import sys, os, subprocess, json, tempfile

PYTHON = r"C:\Users\mikda\AppData\Local\Programs\Python\Python39\python.exe"
DIR = os.path.dirname(os.path.abspath(__file__))

KDB_SCRIPT = r'''
import sys, os, numpy as np, time, json
sys.path.insert(0, os.path.dirname(__file__) or '.')
from numba import njit

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

e = int(sys.argv[1])
n = 1 << e
np.random.seed(42)
keys = np.random.randint(0, 2**32, size=n, dtype=np.uint32)
_kdb_radix_sort_uint32(keys[:1024])  # JIT warmup
_kdb_radix_sort_uint32(keys)         # data warmup
t0 = time.perf_counter()
_kdb_radix_sort_uint32(keys)
dt = time.perf_counter() - t0
with open(sys.argv[2], 'w') as f:
    f.write(json.dumps({"time": dt, "n": n}))
'''

FS_SCRIPT = r'''
import sys, os, numpy as np, time, json
from fractalsort_core.frmw_io_fast import (frmw_io_fast_process, build_counters_from_hist,
                                           _build_reverse_lut, _bit_reverse_lut32)
from numba import njit

p = 32; cache_mb = 8; wc_margin = 2; n_batches = 4
lb = int(sys.argv[3]) if len(sys.argv) > 3 else 10

@njit(nogil=True, fastmath=True)
def precount_bins_fast(keys, ln_minus_lb, lut):
    n_bins = np.int64(1) << ln_minus_lb
    mask = np.int64(n_bins - 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(keys.size):
        bid = _bit_reverse_lut32(np.int64(keys[i]) & mask, ln_minus_lb, lut)
        counts[bid] += 1
    return counts

e = int(sys.argv[1])
n = 1 << e
ln = e; ln_minus_lb = ln - lb
n_io_bins = 1 << ln_minus_lb
batch_size = n // n_batches

# Build layout
n_levels = ln + 1
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
if lc < 1: lc = 1
widths_c = widths[:lc].copy()
per_word_c = per_word[:lc].copy()
word_offsets_c = np.zeros(lc, dtype=np.int64)
for l in range(1, lc):
    word_offsets_c[l] = word_offsets_c[l - 1] + n_words[l - 1]
total_words = int(word_offsets_c[-1] + n_words[lc - 1]) if lc > 0 else 1

lut = _build_reverse_lut()
np.random.seed(42)
keys = np.random.randint(0, 2**32, size=n, dtype=np.uint32)

all_bc = precount_bins_fast(keys, ln_minus_lb, lut)
bin_starts = np.zeros(n_io_bins, dtype=np.int64)
for i in range(1, n_io_bins):
    bin_starts[i] = bin_starts[i - 1] + all_bc[i - 1]

# Warmup (1 run)
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

# Timed (1 run)
hist2 = np.zeros(n_io_bins, dtype=np.int64)
sbatch_mem2 = np.empty(n, dtype=np.uint32)
bin_counts2 = np.zeros(n_io_bins, dtype=np.int32)
bin_wp2 = bin_starts.copy()
t0 = time.perf_counter()
for b in range(n_batches):
    i0, i1 = b * batch_size, (b + 1) * batch_size
    frmw_io_fast_process(keys[i0:i1], hist2, ln_minus_lb, lb, p, ln,
                         sbatch_mem2, bin_wp2, bin_counts2, lut)
cnts2 = np.zeros(total_words, dtype=np.uint64)
build_counters_from_hist(hist2, cnts2, widths_c, per_word_c, word_offsets_c, lc, n_io_bins)
dt = time.perf_counter() - t0

with open(sys.argv[2], 'w') as f:
    f.write(json.dumps({"time": dt, "n": n, "lb": lb, "n_io_bins": n_io_bins}))
'''


def run_sample(script_code, e, extra_args=None):
    """Run script in subprocess, read result from temp file."""
    tmp_py = os.path.join(DIR, f'_tmp_bench_{os.getpid()}.py')
    tmp_out = tmp_py + '.json'
    try:
        with open(tmp_py, 'w') as f:
            f.write(script_code)
        cmd = [PYTHON, tmp_py, str(e), tmp_out]
        if extra_args:
            cmd.extend(str(a) for a in extra_args)
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=DIR, timeout=300)
        if os.path.exists(tmp_out):
            with open(tmp_out) as f:
                return json.loads(f.read()), None
        return None, f"rc={r.returncode}"
    finally:
        for p in (tmp_py, tmp_out):
            if os.path.exists(p):
                os.unlink(p)


if __name__ == '__main__':
    e = int(sys.argv[1])
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    lb_override = int(sys.argv[3]) if len(sys.argv) > 3 else None
    n = 1 << e

    lb_label = f"lb={lb_override}" if lb_override else "lb=10(default)"
    print(f"Benchmarking e={e}, n={n:,} ({n_samples} samples, {lb_label})")
    print()

    # kdb+ samples
    kdb_times = []
    for i in range(n_samples):
        result, err = run_sample(KDB_SCRIPT, e)
        if result:
            kdb_times.append(result['time'])
            print(f"  kdb+ sample {i}: {result['time']*1e6:.0f} us")
        else:
            print(f"  kdb+ sample {i}: FAILED ({err})")
    if not kdb_times:
        print("kdb+ all failed"); sys.exit(1)

    # FS samples
    fs_times = []
    fs_extra = [lb_override] if lb_override else []
    for i in range(n_samples):
        result, err = run_sample(FS_SCRIPT, e, fs_extra)
        if result:
            fs_times.append(result['time'])
            lb = result.get('lb', '?')
            bins = result.get('n_io_bins', '?')
            print(f"  FS   sample {i}: {result['time']*1e6:.0f} us")
        else:
            print(f"  FS   sample {i}: FAILED ({err})")
    if not fs_times:
        print("FS all failed"); sys.exit(1)

    t_kdb = min(kdb_times)
    t_fs = min(fs_times)
    kdb_mks = n / t_kdb / 1e6
    fs_mks = n / t_fs / 1e6
    speedup = fs_mks / kdb_mks
    win = '<-- FS wins' if speedup > 1.0 else ''

    print()
    print(f"e={e} n={n:>12,}  kdb+={kdb_mks:>7.1f}M/s  FS={fs_mks:>7.1f}M/s  speedup={speedup:.2f}x  {win}")
    print(f"  (lb={lb}, bins={bins}, best-of-{len(kdb_times)} kdb+, best-of-{len(fs_times)} FS)")
