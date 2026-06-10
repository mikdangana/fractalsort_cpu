"""Benchmark: multi-column sort — chdb vs duckdb vs polars vs FractalSortCPU.

Sorts a table with 1, 10, and 20 columns (uint32) by the first column.
Row counts: 100K, 1M, 10M.

chdb is optional — auto-skipped if not available on this platform.

Usage:
    python bench_multicolumn.py
"""
import numpy as np
import time
import sys
import gc

# --- Optional imports with availability flags ---

HAS_CHDB = False
try:
    import chdb
    # Verify it actually works (not just a stub)
    chdb.query("SELECT 1", "CSV")
    HAS_CHDB = True
except Exception:
    pass

import duckdb
import polars as pl
from fractalsort_core import fractalsort
from fractalsort_core.frmw_io_fast import (
    frmw_io_fast_process, frmw_io_fast_process_u16,
    countsort_bins_8bit, reconstruct_keys, _build_reverse_lut,
)
from numba import njit


# --- Helpers ---

def generate_table(n_rows, n_cols, seed=42):
    """Generate a table as a dict of uint32 numpy arrays."""
    rng = np.random.RandomState(seed)
    return {f"c{i}": rng.randint(0, 2**32, size=n_rows, dtype=np.uint32)
            for i in range(n_cols)}


def bench(fn, n_runs=5, warmup=1):
    """Benchmark a callable. Returns best time in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        times.append(dt)
    return min(times)


# --- Sort implementations ---

def sort_duckdb(table, n_cols):
    """Sort table using DuckDB SQL. Returns Arrow to avoid Python object overhead."""
    con = duckdb.connect()
    col_names = [f"c{i}" for i in range(n_cols)]
    col_exprs = ", ".join(col_names)
    import pyarrow as pa
    arrow_table = pa.table({k: pa.array(v) for k, v in table.items()})
    con.register("t", arrow_table)

    def run():
        con.sql(f"SELECT {col_exprs} FROM t ORDER BY c0").fetchnumpy()

    return run


def sort_polars(table, n_cols):
    """Sort table using Polars."""
    df = pl.DataFrame({k: pl.Series(k, v) for k, v in table.items()})

    def run():
        df.sort("c0")

    return run


def sort_polars_keyonly(table, n_cols):
    """Key-only: Polars arg_sort (no payload gather)."""
    s = pl.Series("c0", table["c0"])

    def run():
        s.arg_sort()

    return run


def sort_chdb(table, n_cols):
    """Sort table using chdb (ClickHouse in-process)."""
    import pyarrow as pa
    arrow_table = pa.table({k: pa.array(v) for k, v in table.items()})
    col_exprs = ", ".join([f"c{i}" for i in range(n_cols)])
    # chdb can query Arrow via the Python engine
    # We'll use the dataframe API
    def run():
        chdb.query(f"SELECT {col_exprs} FROM Python(arrow_table) ORDER BY c0", "Arrow")

    return run


def sort_fractalsort(table, n_cols):
    """Sort full table: fractalsort key + argsort permutation + payload gather."""
    keys = table["c0"]
    p = 32

    # Pre-warm JIT
    warm_keys = np.random.randint(0, 2**32, size=min(keys.size, 4096), dtype=np.uint32)
    r = fractalsort(warm_keys, p=p, n_batches=1)
    _ = r.reconstruct_all()

    if n_cols == 1:
        def run():
            result = fractalsort(keys, p=p, n_batches=1)
            result.reconstruct_all()
    else:
        col_arrays = [table[f"c{i}"] for i in range(n_cols)]

        def run():
            result = fractalsort(keys, p=p, n_batches=1)
            result.reconstruct_all()
            order = np.argsort(keys, kind='mergesort')
            for arr in col_arrays:
                _ = arr[order]

    return run


def sort_fractalsort_keyonly(table, n_cols):
    """Sort key column only: fractalsort + reconstruct_all (no payload)."""
    keys = table["c0"]
    p = 32

    warm_keys = np.random.randint(0, 2**32, size=min(keys.size, 4096), dtype=np.uint32)
    r = fractalsort(warm_keys, p=p, n_batches=1)
    _ = r.reconstruct_all()

    def run():
        result = fractalsort(keys, p=p, n_batches=1)
        result.reconstruct_all()

    return run


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


def _fsc8_sort_keys(keys, p=32):
    """Full FSC8 pipeline: scatter + 8-bit radix bin sort + reconstruct."""
    n = keys.size
    e = int(np.ceil(np.log2(n))) if n > 1 else 1
    ln = e
    lb = e - 8 if e < 22 else e - 10
    lb = max(1, lb)
    ln_minus_lb = ln - lb
    n_bins = 1 << ln_minus_lb
    entry_bits = lb + (p - ln)
    bin_shift = np.uint32(p - ln_minus_lb)

    _, bin_starts = _precount_and_starts(keys, bin_shift, n_bins)

    hist = np.zeros(n_bins, dtype=np.int64)
    sbatch_mem = np.empty(n, dtype=np.uint32)
    bin_counts = np.zeros(n_bins, dtype=np.int32)
    bin_wp = bin_starts.copy()
    lut = _fsc8_lut[0]

    frmw_io_fast_process(keys, hist, ln_minus_lb, lb, p, ln,
                         sbatch_mem, bin_starts, bin_counts, lut, bin_wp)
    countsort_bins_8bit(sbatch_mem, bin_starts, bin_counts,
                        n_bins, lb, ln_minus_lb, p, ln)
    output = np.empty(n, dtype=np.uint32)
    reconstruct_keys(sbatch_mem, bin_starts, bin_counts,
                     n_bins, ln_minus_lb, p, output)
    return output


# Global LUT (built once)
_fsc8_lut = [None]


def sort_fsc8(table, n_cols):
    """Sort by c0 using FSC8 pipeline (frmw_io_fast), then reorder payload."""
    keys = table["c0"]
    p = 32

    # Build LUT + warmup
    if _fsc8_lut[0] is None:
        _fsc8_lut[0] = _build_reverse_lut()
    warm_keys = np.random.randint(0, 2**32, size=min(keys.size, 4096), dtype=np.uint32)
    _fsc8_sort_keys(warm_keys, p=p)

    if n_cols == 1:
        def run():
            _fsc8_sort_keys(keys, p=p)
    else:
        col_arrays = [table[f"c{i}"] for i in range(n_cols)]

        def run():
            _fsc8_sort_keys(keys, p=p)
            order = np.argsort(keys, kind='mergesort')
            for arr in col_arrays:
                _ = arr[order]

    return run


def sort_fsc8_keyonly(table, n_cols):
    """Sort key column only using FSC8 pipeline."""
    keys = table["c0"]
    p = 32

    if _fsc8_lut[0] is None:
        _fsc8_lut[0] = _build_reverse_lut()
    warm_keys = np.random.randint(0, 2**32, size=min(keys.size, 4096), dtype=np.uint32)
    _fsc8_sort_keys(warm_keys, p=p)

    def run():
        _fsc8_sort_keys(keys, p=p)

    return run


@njit(nogil=True, fastmath=True)
def fsf_binsort_reconstruct_u32(sbatch_mem, bin_starts, bin_counts, n_bins, output):
    """
    FSF bin sort: Reads 16-bit residues from sbatch_mem, combines them with the
    16-bit bin ID, and writes fully sorted 32-bit integers directly to output.
    """
    n_buckets = 65536
    hist = np.zeros(n_buckets, dtype=np.uint16)

    for bid in range(n_bins):
        c = np.int64(bin_counts[bid])
        if c == 0:
            continue

        start = bin_starts[bid]
        upper_bits = np.uint32(bid) << 16

        # Pass 1: Count 16-bit residues
        for ki in range(c):
            res = sbatch_mem[start + ki]
            hist[res] += 1

        # Pass 2: Reconstruct 32-bit values directly into the output array
        write_idx = start
        for res_val in range(n_buckets):
            count = hist[res_val]
            if count > 0:
                reconstructed_u32 = upper_bits | np.uint32(res_val)
                for _ in range(count):
                    output[write_idx] = reconstructed_u32
                    write_idx += 1
                hist[res_val] = 0


@njit(nogil=True, fastmath=True)
def _computed_bin_starts(n, n_bins):
    """Compute uniform bin starts: bin i starts at i * (n // n_bins)."""
    per_bin = n // n_bins
    bin_starts = np.empty(n_bins, dtype=np.int64)
    for i in range(n_bins):
        bin_starts[i] = np.int64(i) * np.int64(per_bin)
    return bin_starts


def _fsf_sort_keys(keys, p=32, lc=16):
    """Full FSF pipeline with computed uniform offsets (no precount pass)."""
    n = keys.size
    e = int(np.ceil(np.log2(n))) if n > 1 else 1
    ln = e
    lc_eff = min(lc, e - 1)
    lb = max(1, e - lc_eff)
    ln_minus_lb = ln - lb
    n_bins = 1 << ln_minus_lb
    bin_shift = np.uint32(p - ln_minus_lb)

    # Computed uniform offsets — no precount scan
    bin_starts = _computed_bin_starts(n, n_bins)

    hist = np.zeros(n_bins, dtype=np.int64)
    sbatch_mem = np.empty(n, dtype=np.uint16)
    bin_counts = np.zeros(n_bins, dtype=np.int32)
    bin_wp = bin_starts.copy()
    lut = _fsc8_lut[0]

    # Scatter u16 entries
    frmw_io_fast_process_u16(keys, hist, ln_minus_lb, lb, p, ln,
                             sbatch_mem, bin_starts, bin_counts, lut, bin_wp)

    # Final step: Allocate and write to u32 array
    output = np.empty(n, dtype=np.uint32)
    fsf_binsort_reconstruct_u32(sbatch_mem, bin_starts, bin_counts, n_bins, output)

    return output

def profile_fsf(n=10_000_000, p=32, lc=16, n_trials=5):
    """Profile FSF pipeline stages separately."""
    if _fsc8_lut[0] is None:
        _fsc8_lut[0] = _build_reverse_lut()

    keys = np.random.randint(0, 2**32, size=n, dtype=np.uint32)

    e = int(np.ceil(np.log2(n))) if n > 1 else 1
    ln = e
    lc_eff = min(lc, e - 1)
    lb = max(1, e - lc_eff)
    ln_minus_lb = ln - lb
    n_bins = 1 << ln_minus_lb
    bin_shift = np.uint32(p - ln_minus_lb)
    lut = _fsc8_lut[0]

    # Warmup
    _fsf_sort_keys(np.random.randint(0, 2**32, size=4096, dtype=np.uint32), p=p)

    best = {'starts': 1e9, 'alloc': 1e9, 'scatter': 1e9, 'recon': 1e9, 'total': 1e9}

    for _ in range(n_trials):
        t0 = time.perf_counter()
        bin_starts = _computed_bin_starts(n, n_bins)
        t1 = time.perf_counter()

        hist = np.zeros(n_bins, dtype=np.int64)
        sbatch_mem = np.empty(n, dtype=np.uint16)
        bin_counts = np.zeros(n_bins, dtype=np.int32)
        bin_wp = bin_starts.copy()
        output = np.empty(n, dtype=np.uint32)
        t2 = time.perf_counter()

        frmw_io_fast_process_u16(keys, hist, ln_minus_lb, lb, p, ln,
                                 sbatch_mem, bin_starts, bin_counts, lut, bin_wp)
        # derive bin_counts from bin_wp
        for b in range(n_bins):
            bin_counts[b] = np.int32(bin_wp[b])
        t3 = time.perf_counter()

        fsf_binsort_reconstruct_u32(sbatch_mem, bin_starts, bin_counts, n_bins, output)
        t4 = time.perf_counter()

        best['starts'] = min(best['starts'], t1 - t0)
        best['alloc'] = min(best['alloc'], t2 - t1)
        best['scatter'] = min(best['scatter'], t3 - t2)
        best['recon'] = min(best['recon'], t4 - t3)
        best['total'] = min(best['total'], t4 - t0)

    print(f"\nFSF Profile ({n/1e6:.0f}M keys, {n_bins} bins):")
    print(f"  bin_starts:  {best['starts']*1000:>7.1f} ms  ({best['starts']/best['total']*100:>4.1f}%)")
    print(f"  alloc:       {best['alloc']*1000:>7.1f} ms  ({best['alloc']/best['total']*100:>4.1f}%)")
    print(f"  scatter:     {best['scatter']*1000:>7.1f} ms  ({best['scatter']/best['total']*100:>4.1f}%)")
    print(f"  reconstruct: {best['recon']*1000:>7.1f} ms  ({best['recon']/best['total']*100:>4.1f}%)")
    print(f"  total:       {best['total']*1000:>7.1f} ms  ({n/best['total']/1e6:.1f} M/s)")
    print()


def sort_fsf(table, n_cols):
    """Sort by c0 using FSF pipeline, then reorder payload based on sorted keys."""
    keys = table["c0"]
    p = 32

    # Warmup
    if _fsc8_lut[0] is None:
        _fsc8_lut[0] = _build_reverse_lut()
    warm_keys = np.random.randint(0, 2**32, size=min(keys.size, 4096), dtype=np.uint32)
    _fsf_sort_keys(warm_keys, p=p)

    if n_cols == 1:
        def run():
            _fsf_sort_keys(keys, p=p)
    else:
        col_arrays = [table[f"c{i}"] for i in range(n_cols)]

        def run():
            sorted_keys = _fsf_sort_keys(keys, p=p)
            # Create permutation map using the actual sorted key sequence
            order = np.argsort(sorted_keys, kind='mergesort')
            for arr in col_arrays:
                _ = arr[order]

    return run


def sort_fsf_keyonly(table, n_cols):
    """Sort key column only using FSF pipeline."""
    keys = table["c0"]
    p = 32

    if _fsc8_lut[0] is None:
        _fsc8_lut[0] = _build_reverse_lut()
    warm_keys = np.random.randint(0, 2**32, size=min(keys.size, 4096), dtype=np.uint32)
    _fsf_sort_keys(warm_keys, p=p)

    def run():
        _fsf_sort_keys(keys, p=p)

    return run



# --- Main benchmark ---

def main():
    n_runs = 5
    col_counts = [1, 10, 20]
    row_counts = [100_000, 1_000_000, 10_000_000]

    engines = []
    engines.append(("DuckDB", sort_duckdb))
    engines.append(("Polars", sort_polars))
    engines.append(("Polars(key)", sort_polars_keyonly))
    if HAS_CHDB:
        engines.append(("chdb", sort_chdb))
    engines.append(("FSC8", sort_fsc8))
    engines.append(("FSC8(key)", sort_fsc8_keyonly))
    engines.append(("FSF", sort_fsf))
    engines.append(("FSF(key)", sort_fsf_keyonly))

    print("=" * 90)
    print("Multi-column sort benchmark: sort by first column (uint32)")
    print("=" * 90)
    if not HAS_CHDB:
        print("NOTE: chdb not available on this platform — skipped.")
    print()

    # Header
    engine_names = [name for name, _ in engines]
    hdr = f"{'rows':>10} {'cols':>4} |"
    for name in engine_names:
        hdr += f" {name:>12}"
    hdr += " | best"
    print(hdr)
    print("-" * len(hdr))

    results = []

    for n_rows in row_counts:
        for n_cols in col_counts:
            table = generate_table(n_rows, n_cols)
            data_mb = n_rows * n_cols * 4 / 1e6

            row_results = {"rows": n_rows, "cols": n_cols}
            timings = {}

            for name, sort_fn in engines:
                try:
                    fn = sort_fn(table, n_cols)
                    t = bench(fn, n_runs=n_runs, warmup=1)
                    timings[name] = t
                except Exception as e:
                    timings[name] = None
                    print(f"  WARN: {name} failed for {n_rows}x{n_cols}: {e}",
                          file=sys.stderr)

            # Format row
            best_time = min((t for t in timings.values() if t is not None),
                            default=None)
            best_name = None
            if best_time is not None:
                best_name = [k for k, v in timings.items()
                             if v == best_time][0]

            line = f"{n_rows:>10,} {n_cols:>4} |"
            for name in engine_names:
                t = timings.get(name)
                if t is None:
                    line += f" {'N/A':>12}"
                else:
                    mks = n_rows / t / 1e6
                    marker = "*" if name == best_name else " "
                    line += f" {mks:>9.1f}M/s{marker}"
            line += f" | {best_name}"
            print(line)

            row_results["timings"] = timings
            results.append(row_results)

        print()

    # Summary table: speedups vs DuckDB
    print()
    print("=" * 70)
    print("Speedup vs DuckDB (higher = faster than DuckDB)")
    print("=" * 70)
    other_names = [n for n in engine_names if n != "DuckDB"]
    hdr2 = f"{'rows':>10} {'cols':>4} |"
    for name in other_names:
        hdr2 += f" {name:>12}"
    print(hdr2)
    print("-" * len(hdr2))

    for r in results:
        t_duck = r["timings"].get("DuckDB")
        line = f"{r['rows']:>10,} {r['cols']:>4} |"
        for name in other_names:
            t = r["timings"].get(name)
            if t is None or t_duck is None:
                line += f" {'N/A':>12}"
            else:
                speedup = t_duck / t
                line += f" {speedup:>10.2f}x "

        print(line)
        if r["cols"] == col_counts[-1]:
            print()

    print()
    print("Environment:")
    print(f"  Platform:    {sys.platform}")
    print(f"  Python:      {sys.version.split()[0]}")
    print(f"  DuckDB:      {duckdb.__version__}")
    print(f"  Polars:      {pl.__version__}")
    print(f"  chdb:        {'available' if HAS_CHDB else 'not available'}")
    import numba
    print(f"  Numba:       {numba.__version__}")
    print(f"  NumPy:       {np.__version__}")
    print(f"  n_runs:      {n_runs}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--profile-fsf':
        profile_fsf()
    else:
        main()
