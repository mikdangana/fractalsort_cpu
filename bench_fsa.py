"""Benchmark FS (scatter only) and FSA (scatter + reconstruct) vs kdb+.
Usage: python bench_fsa.py <e> [n_samples] [lb]
"""
import sys, os, subprocess, json

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
_kdb_radix_sort_uint32(keys[:1024])
_kdb_radix_sort_uint32(keys)
t0 = time.perf_counter()
_kdb_radix_sort_uint32(keys)
dt = time.perf_counter() - t0
with open(sys.argv[2], 'w') as f:
    f.write(json.dumps({"time": dt, "n": n}))
'''

COMMON_SETUP = r'''
import sys, os, numpy as np, time, json
from fractalsort_core.frmw_io_fast import (frmw_io_fast_process, frmw_io_fast_process_wc,
                           countsort_bins, countsort_bins_8bit,
                           unpack_wc, reconstruct_keys, reconstruct_sorted,
                           strip_to_trailing, _build_reverse_lut)
from numba import njit

p = 32; n_batches = 4
lb = int(sys.argv[3]) if len(sys.argv) > 3 else 10
sort_k = int(sys.argv[4]) if len(sys.argv) > 4 else 0

@njit(nogil=True, fastmath=True)
def precount_bins(keys, ln_minus_lb, p):
    n_bins = np.int64(1) << ln_minus_lb
    bin_shift = np.uint32(p - ln_minus_lb)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(keys.size):
        bid = np.int64(keys[i] >> bin_shift)
        counts[bid] += 1
    return counts

e = int(sys.argv[1])
n = 1 << e
ln = e; ln_minus_lb = ln - lb
n_io_bins = 1 << ln_minus_lb
batch_size = n // n_batches
lut = _build_reverse_lut()
np.random.seed(42)
keys = np.random.randint(0, 2**32, size=n, dtype=np.uint32)

all_bc = precount_bins(keys, ln_minus_lb, p)
bin_starts = np.zeros(n_io_bins, dtype=np.int64)
for i in range(1, n_io_bins):
    bin_starts[i] = bin_starts[i - 1] + all_bc[i - 1]
total_slots = int(bin_starts[-1] + all_bc[-1])

# Packed layout for WC: each uint64 word holds up to cap entries
entry_bits = lb + (p - e)
if entry_bits <= 20:
    wc_cap = 3
elif entry_bits <= 30:
    wc_cap = 2
else:
    wc_cap = 1
# Each bin needs ceil(bc[i] / cap) packed words
wc_bin_starts = np.zeros(n_io_bins, dtype=np.int64)
for i in range(1, n_io_bins):
    wc_bin_starts[i] = wc_bin_starts[i - 1] + (int(all_bc[i - 1]) + wc_cap - 1) // wc_cap
wc_total = int(wc_bin_starts[-1] + (int(all_bc[-1]) + wc_cap - 1) // wc_cap)
'''

FS_SCRIPT = COMMON_SETUP + r'''
# Warmup
hist = np.zeros(n_io_bins, dtype=np.int64)
sbatch_mem = np.empty(total_slots, dtype=np.uint32)
bin_counts = np.zeros(n_io_bins, dtype=np.int32)
bin_wp = bin_starts.copy()
for b in range(n_batches):
    i0, i1 = b * batch_size, (b + 1) * batch_size
    frmw_io_fast_process(keys[i0:i1], hist, ln_minus_lb, lb, p, ln,
                         sbatch_mem, bin_starts, bin_counts, lut, bin_wp)

# Timed
hist2 = np.zeros(n_io_bins, dtype=np.int64)
sbatch_mem2 = np.empty(total_slots, dtype=np.uint32)
bin_counts2 = np.zeros(n_io_bins, dtype=np.int32)
bin_wp2 = bin_starts.copy()
t0 = time.perf_counter()
for b in range(n_batches):
    i0, i1 = b * batch_size, (b + 1) * batch_size
    frmw_io_fast_process(keys[i0:i1], hist2, ln_minus_lb, lb, p, ln,
                         sbatch_mem2, bin_starts, bin_counts2, lut, bin_wp2)
dt = time.perf_counter() - t0

with open(sys.argv[2], 'w') as f:
    f.write(json.dumps({"time": dt, "n": n, "lb": lb, "n_io_bins": n_io_bins}))
'''

FS_WC_SCRIPT = COMMON_SETUP + r'''
# Warmup
hist = np.zeros(n_io_bins, dtype=np.int64)
sbatch_mem_p = np.empty(wc_total, dtype=np.uint64)
bin_counts = np.zeros(n_io_bins, dtype=np.int32)
bin_wp = wc_bin_starts.copy()
for b in range(n_batches):
    i0, i1 = b * batch_size, (b + 1) * batch_size
    frmw_io_fast_process_wc(keys[i0:i1], hist, ln_minus_lb, lb, p, ln,
                            sbatch_mem_p, wc_bin_starts, bin_counts, lut, bin_wp)

# Timed
hist2 = np.zeros(n_io_bins, dtype=np.int64)
sbatch_mem_p2 = np.empty(wc_total, dtype=np.uint64)
bin_counts2 = np.zeros(n_io_bins, dtype=np.int32)
bin_wp2 = wc_bin_starts.copy()
t0 = time.perf_counter()
for b in range(n_batches):
    i0, i1 = b * batch_size, (b + 1) * batch_size
    frmw_io_fast_process_wc(keys[i0:i1], hist2, ln_minus_lb, lb, p, ln,
                            sbatch_mem_p2, wc_bin_starts, bin_counts2, lut, bin_wp2)
dt = time.perf_counter() - t0

with open(sys.argv[2], 'w') as f:
    f.write(json.dumps({"time": dt, "n": n, "lb": lb, "n_io_bins": n_io_bins}))
'''

FSWS_SCRIPT = COMMON_SETUP + r'''
use_wc = n_io_bins > 1024

# Warmup
hist = np.zeros(n_io_bins, dtype=np.int64)
bin_counts = np.zeros(n_io_bins, dtype=np.int32)
if use_wc:
    sbatch_mem_p = np.empty(wc_total, dtype=np.uint64)
    bin_wp = wc_bin_starts.copy()
    for b in range(n_batches):
        i0, i1 = b * batch_size, (b + 1) * batch_size
        frmw_io_fast_process_wc(keys[i0:i1], hist, ln_minus_lb, lb, p, ln,
                                sbatch_mem_p, wc_bin_starts, bin_counts, lut, bin_wp)
    sbatch_mem = np.empty(total_slots, dtype=np.uint32)
    bc_tmp = np.zeros(n_io_bins, dtype=np.int32)
    unpack_wc(sbatch_mem_p, wc_bin_starts, bin_wp,
              sbatch_mem, bin_starts, bc_tmp, n_io_bins, entry_bits)
else:
    sbatch_mem = np.empty(total_slots, dtype=np.uint32)
    bin_wp = bin_starts.copy()
    for b in range(n_batches):
        i0, i1 = b * batch_size, (b + 1) * batch_size
        frmw_io_fast_process(keys[i0:i1], hist, ln_minus_lb, lb, p, ln,
                             sbatch_mem, bin_starts, bin_counts, lut, bin_wp)
countsort_bins_8bit(sbatch_mem, bin_starts, bin_counts,
                    n_io_bins, lb, ln_minus_lb, p, ln)

# Timed
hist2 = np.zeros(n_io_bins, dtype=np.int64)
bin_counts2 = np.zeros(n_io_bins, dtype=np.int32)
if use_wc:
    sbatch_mem_p2 = np.empty(wc_total, dtype=np.uint64)
    bin_wp2 = wc_bin_starts.copy()
    t0 = time.perf_counter()
    for b in range(n_batches):
        i0, i1 = b * batch_size, (b + 1) * batch_size
        frmw_io_fast_process_wc(keys[i0:i1], hist2, ln_minus_lb, lb, p, ln,
                                sbatch_mem_p2, wc_bin_starts, bin_counts2, lut, bin_wp2)
    sbatch_mem2 = np.empty(total_slots, dtype=np.uint32)
    bc_tmp2 = np.zeros(n_io_bins, dtype=np.int32)
    unpack_wc(sbatch_mem_p2, wc_bin_starts, bin_wp2,
              sbatch_mem2, bin_starts, bc_tmp2, n_io_bins, entry_bits)
    countsort_bins_8bit(sbatch_mem2, bin_starts, bin_counts2,
                        n_io_bins, lb, ln_minus_lb, p, ln)
    dt = time.perf_counter() - t0
else:
    sbatch_mem2 = np.empty(total_slots, dtype=np.uint32)
    bin_wp2 = bin_starts.copy()
    t0 = time.perf_counter()
    for b in range(n_batches):
        i0, i1 = b * batch_size, (b + 1) * batch_size
        frmw_io_fast_process(keys[i0:i1], hist2, ln_minus_lb, lb, p, ln,
                             sbatch_mem2, bin_starts, bin_counts2, lut, bin_wp2)
    countsort_bins_8bit(sbatch_mem2, bin_starts, bin_counts2,
                        n_io_bins, lb, ln_minus_lb, p, ln)
    dt = time.perf_counter() - t0

with open(sys.argv[2], 'w') as f:
    f.write(json.dumps({"time": dt, "n": n, "lb": lb, "n_io_bins": n_io_bins}))
'''

FSC_SCRIPT = COMMON_SETUP + r'''
# Warmup
hist = np.zeros(n_io_bins, dtype=np.int64)
sbatch_mem = np.empty(total_slots, dtype=np.uint32)
bin_counts = np.zeros(n_io_bins, dtype=np.int32)
bin_wp = bin_starts.copy()
for b in range(n_batches):
    i0, i1 = b * batch_size, (b + 1) * batch_size
    frmw_io_fast_process(keys[i0:i1], hist, ln_minus_lb, lb, p, ln,
                         sbatch_mem, bin_starts, bin_counts, lut, bin_wp)
strip_to_trailing(sbatch_mem, bin_starts, bin_counts, n_io_bins, lb, p, ln)

# Timed: scatter + strip to trailing + persist
hist2 = np.zeros(n_io_bins, dtype=np.int64)
sbatch_mem2 = np.empty(total_slots, dtype=np.uint32)
bin_counts2 = np.zeros(n_io_bins, dtype=np.int32)
bin_wp2 = bin_starts.copy()
t0 = time.perf_counter()
for b in range(n_batches):
    i0, i1 = b * batch_size, (b + 1) * batch_size
    frmw_io_fast_process(keys[i0:i1], hist2, ln_minus_lb, lb, p, ln,
                         sbatch_mem2, bin_starts, bin_counts2, lut, bin_wp2)
strip_to_trailing(sbatch_mem2, bin_starts, bin_counts2, n_io_bins, lb, p, ln)
dt = time.perf_counter() - t0

with open(sys.argv[2], 'w') as f:
    f.write(json.dumps({"time": dt, "n": n, "lb": lb, "n_io_bins": n_io_bins}))
'''

FSA_SCRIPT = COMMON_SETUP + r'''
# Warmup
hist = np.zeros(n_io_bins, dtype=np.int64)
sbatch_mem = np.empty(total_slots, dtype=np.uint32)
bin_counts = np.zeros(n_io_bins, dtype=np.int32)
bin_wp = bin_starts.copy()
for b in range(n_batches):
    i0, i1 = b * batch_size, (b + 1) * batch_size
    frmw_io_fast_process(keys[i0:i1], hist, ln_minus_lb, lb, p, ln,
                         sbatch_mem, bin_starts, bin_counts, lut, bin_wp)
countsort_bins(sbatch_mem, bin_starts, bin_counts,
               n_io_bins, lb, ln_minus_lb, p, ln)
output = np.empty(n, dtype=np.uint32)
reconstruct_keys(sbatch_mem, bin_starts, bin_counts,
                 n_io_bins, ln_minus_lb, p, output)

# Timed: scatter + count-sort + reconstruct full keys
hist2 = np.zeros(n_io_bins, dtype=np.int64)
sbatch_mem2 = np.empty(total_slots, dtype=np.uint32)
bin_counts2 = np.zeros(n_io_bins, dtype=np.int32)
bin_wp2 = bin_starts.copy()
output2 = np.empty(n, dtype=np.uint32)
t0 = time.perf_counter()
for b in range(n_batches):
    i0, i1 = b * batch_size, (b + 1) * batch_size
    frmw_io_fast_process(keys[i0:i1], hist2, ln_minus_lb, lb, p, ln,
                         sbatch_mem2, bin_starts, bin_counts2, lut, bin_wp2)
countsort_bins(sbatch_mem2, bin_starts, bin_counts2,
               n_io_bins, lb, ln_minus_lb, p, ln)
reconstruct_keys(sbatch_mem2, bin_starts, bin_counts2,
                 n_io_bins, ln_minus_lb, p, output2)
dt = time.perf_counter() - t0

with open(sys.argv[2], 'w') as f:
    f.write(json.dumps({"time": dt, "n": n, "lb": lb, "n_io_bins": n_io_bins}))
'''


def run_sample(script_code, e, extra_args=None):
    tmp_py = os.path.join(DIR, f'_tmp_bench_{os.getpid()}.py')
    tmp_out = tmp_py + '.json'
    try:
        with open(tmp_py, 'w') as f:
            f.write(script_code)
        cmd = [PYTHON, tmp_py, str(e), tmp_out]
        if extra_args:
            cmd.extend(str(a) for a in extra_args)
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=DIR, timeout=600)
        if os.path.exists(tmp_out):
            with open(tmp_out) as f:
                return json.loads(f.read()), None
        return None, f"rc={r.returncode} stderr={r.stderr[-200:]}"
    finally:
        for p in (tmp_py, tmp_out):
            if os.path.exists(p):
                os.unlink(p)


if __name__ == '__main__':
    e = int(sys.argv[1])
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    lb_override = int(sys.argv[3]) if len(sys.argv) > 3 else None
    n = 1 << e

    lb_val = lb_override if lb_override else 10
    if lb_override is None and e > 20:
        lb_val = e - 8
    sort_k = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    extra = [lb_val, sort_k]

    trailing = 32 - e
    sort_bits = lb_val + min(sort_k, trailing)
    print(f"e={e}, n={n:,}, lb={lb_val}, bins={1<<(e-lb_val)}, k={sort_k}, sort_bits={sort_bits}, counters={1<<sort_bits}, {n_samples} samples")
    print()

    kdb_times = []
    for i in range(n_samples):
        result, err = run_sample(KDB_SCRIPT, e)
        if result:
            kdb_times.append(result['time'])
            print(f"  kdb+ #{i}: {result['time']*1e6:.0f} us")
        else:
            print(f"  kdb+ #{i}: FAILED ({err})")

    fs_times = []
    for i in range(n_samples):
        result, err = run_sample(FS_SCRIPT, e, extra)
        if result:
            fs_times.append(result['time'])
            print(f"  FS   #{i}: {result['time']*1e6:.0f} us")
        else:
            print(f"  FS   #{i}: FAILED ({err})")

    fswc_times = []
    for i in range(n_samples):
        result, err = run_sample(FS_WC_SCRIPT, e, extra)
        if result:
            fswc_times.append(result['time'])
            print(f"  FSWC #{i}: {result['time']*1e6:.0f} us")
        else:
            print(f"  FSWC #{i}: FAILED ({err})")

    fsws_times = []
    for i in range(n_samples):
        result, err = run_sample(FSWS_SCRIPT, e, extra)
        if result:
            fsws_times.append(result['time'])
            print(f"  FSWS #{i}: {result['time']*1e6:.0f} us")
        else:
            print(f"  FSWS #{i}: FAILED ({err})")

    fsc_times = []
    for i in range(n_samples):
        result, err = run_sample(FSC_SCRIPT, e, extra)
        if result:
            fsc_times.append(result['time'])
            print(f"  FSC  #{i}: {result['time']*1e6:.0f} us")
        else:
            print(f"  FSC  #{i}: FAILED ({err})")

    fsa_times = []
    for i in range(n_samples):
        result, err = run_sample(FSA_SCRIPT, e, extra)
        if result:
            fsa_times.append(result['time'])
            print(f"  FSA  #{i}: {result['time']*1e6:.0f} us")
        else:
            print(f"  FSA  #{i}: FAILED ({err})")

    if not kdb_times or not fs_times or not fswc_times or not fsws_times or not fsc_times or not fsa_times:
        print("Some benchmarks failed"); sys.exit(1)

    t_kdb = min(kdb_times)
    t_fs = min(fs_times)
    t_fswc = min(fswc_times)
    t_fsws = min(fsws_times)
    t_fsc = min(fsc_times)
    t_fsa = min(fsa_times)
    kdb_mks = n / t_kdb / 1e6
    fs_mks = n / t_fs / 1e6
    fswc_mks = n / t_fswc / 1e6
    fsws_mks = n / t_fsws / 1e6
    fsc_mks = n / t_fsc / 1e6
    fsa_mks = n / t_fsa / 1e6

    print()
    print(f"e={e:2d}  n={n:>12,}  lb={lb_val}  bins={1<<(e-lb_val)}")
    print(f"  kdb+ = {kdb_mks:>7.1f} M/s  ({t_kdb*1e6:>10.0f} us)")
    print(f"  FS   = {fs_mks:>7.1f} M/s  ({t_fs*1e6:>10.0f} us)  {fs_mks/kdb_mks:.2f}x vs kdb+")
    print(f"  FSWC = {fswc_mks:>7.1f} M/s  ({t_fswc*1e6:>10.0f} us)  {fswc_mks/kdb_mks:.2f}x vs kdb+")
    print(f"  FSWS = {fsws_mks:>7.1f} M/s  ({t_fsws*1e6:>10.0f} us)  {fsws_mks/kdb_mks:.2f}x vs kdb+")
    print(f"  FSC  = {fsc_mks:>7.1f} M/s  ({t_fsc*1e6:>10.0f} us)  {fsc_mks/kdb_mks:.2f}x vs kdb+")
    print(f"  FSA  = {fsa_mks:>7.1f} M/s  ({t_fsa*1e6:>10.0f} us)  {fsa_mks/kdb_mks:.2f}x vs kdb+")
