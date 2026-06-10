"""Benchmark FS8, FSC8, FSF vs kdb+ radix sort.

FS8:  step 2 (scatter) + step 3 (8-bit radix bin sort)
FSC8: step 2 (scatter) + step 3 (8-bit radix bin sort) + step 4 (reconstruct)
FSF:  step 2 (scatter) + step 3 (packed histogram bin sort, 4-bit sparse)
kdb+: LSB radix sort (256-way, 4-pass, 32-bit)

All keys uint32, p=32.
FS8/FSC8: lb = e-8 (e<22) or e-10 (e>=22).
FSF:      lc=16, entry_bits = p-lc = 16, lb = e-lc = e-16.
"""
import numpy as np
import time
import sys

from fractalsort_core.frmw_io_fast import (
    frmw_io_fast_process, frmw_io_fast_process_u16,
    frmw_io_fast_process_u16_dram, countsort_bins_8bit,
    reconstruct_keys, _build_reverse_lut,
)
from numba import njit

p = 32


@njit(nogil=True, fastmath=True)
def kdb_radix_sort(keys):
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
def precount_and_starts(keys, bin_shift, n_bins):
    all_bc = np.zeros(n_bins, dtype=np.int64)
    for i in range(keys.size):
        all_bc[np.int64(keys[i] >> bin_shift)] += 1
    bin_starts = np.zeros(n_bins, dtype=np.int64)
    for i in range(1, n_bins):
        bin_starts[i] = bin_starts[i - 1] + all_bc[i - 1]
    return bin_starts


@njit(nogil=True, fastmath=True)
def fsf_binsort_hist_only(sbatch_mem, bin_starts, bin_counts, n_bins, entry_bits):
    """FSF bin sort: build 4-bit packed histogram per bin. No reconstruction."""
    n_buckets = np.int64(1) << entry_bits
    packed_size = (n_buckets + 1) // 2
    packed_hist = np.zeros(packed_size, dtype=np.uint8)

    for bid in range(n_bins):
        c = np.int64(bin_counts[bid])
        if c == 0:
            continue

        start = bin_starts[bid]

        # Single pass: count entries into packed 4-bit histogram
        for ki in range(c):
            val = np.int64(sbatch_mem[start + ki])
            byte_idx = val >> 1
            if val & 1:
                packed_hist[byte_idx] += np.uint8(0x10)
            else:
                packed_hist[byte_idx] += np.uint8(0x01)

        # Clear only touched locations
        for ki in range(c):
            val = np.int64(sbatch_mem[start + ki])
            packed_hist[val >> 1] = 0



@njit(nogil=True, fastmath=True)
def fsf_binsort_reconstruct_u32(sbatch_mem, bin_starts, bin_counts, n_bins, entry_bits):
    """
    FSF bin sort: Reads 16-bit residues from sbatch_mem, combines them with the 
    16-bit bin ID, and overwrites sbatch_mem with the reconstructed, fully sorted 
    32-bit integers.
    """
    # 2^16 buckets for 16-bit entries
    n_buckets = 65536 
    
    # Local histogram buffer (64KB - fits perfectly in CPU L1 Cache)
    # Using uint16 assumes no single residue repeats > 65535 times per single bin
    hist = np.zeros(n_buckets, dtype=np.uint16)

    # Pre-calculate the upper 16 bits shift
    # If bid is the leading 16 bits, we shift it left by 16
    for bid in range(n_bins):
        c = np.int64(bin_counts[bid])
        if c == 0:
            continue

        start = bin_starts[bid]
        upper_bits = np.uint32(bid) << 16

        # Pass 1: Count occurrences of each 16-bit residue
        for ki in range(c):
            res = sbatch_mem[start + ki]
            hist[res] += 1

        # Pass 2: Reconstruct fully sorted 32-bit values back into sbatch_mem
        write_idx = start
        for res_val in range(n_buckets):
            count = hist[res_val]
            if count > 0:
                # Combine upper 16 bits (bid) and lower 16 bits (res_val)
                reconstructed_u32 = upper_bits | np.uint32(res_val)
                
                # Write it out sequentially
                for _ in range(count):
                    sbatch_mem[write_idx] = reconstructed_u32
                    write_idx += 1
                
                # Clear the histogram slot as we go! (Saves a third loop)
                hist[res_val] = 0



@njit(nogil=True, fastmath=True)
def fsf_binsort_dram(sbatch_mem, bin_counts, n_bins, entry_bits,
                     ldram, nbins):
    """FSF bin sort reading entries in DRAM stride layout.

    Entries for bin `bid` are stored in stripes:
      stripe s: base = s * ldram * nbins + bid * ldram
      within stripe: offsets 0..ldram-1
    Total entries per bin = bin_counts[bid].
    """
    n_buckets = np.int64(1) << entry_bits
    packed_size = (n_buckets + 1) // 2
    packed_hist = np.zeros(packed_size, dtype=np.uint8)
    mem_len = len(sbatch_mem)
    stripe_stride = np.int64(ldram) * np.int64(nbins)

    for bid in range(n_bins):
        c = np.int64(bin_counts[bid])
        if c == 0:
            continue

        bin_base = np.int64(bid) * np.int64(ldram)
        remaining = c
        stripe = np.int64(0)

        # Read entries stripe by stripe, build histogram
        while remaining > 0:
            chunk = min(remaining, np.int64(ldram))
            stripe_base = stripe * stripe_stride + bin_base
            for ki in range(chunk):
                addr = (stripe_base + ki) % mem_len
                val = np.int64(sbatch_mem[addr])
                byte_idx = val >> 1
                if val & 1:
                    packed_hist[byte_idx] += np.uint8(0x10)
                else:
                    packed_hist[byte_idx] += np.uint8(0x01)
            remaining -= chunk
            stripe += 1

        # Clear: re-read same entries
        remaining = c
        stripe = np.int64(0)
        while remaining > 0:
            chunk = min(remaining, np.int64(ldram))
            stripe_base = stripe * stripe_stride + bin_base
            for ki in range(chunk):
                addr = (stripe_base + ki) % mem_len
                packed_hist[np.int64(sbatch_mem[addr]) >> 1] = 0
            remaining -= chunk
            stripe += 1


def run_fsf(keys, e, lut, lc=16):
    """FSF: scatter + 4-bit packed histogram bin sort.

    Defaults: lc=16, entry_bits = p - lc = 16, lb = e - lc.
    These are FSF's own parameters, independent of FS8/FSC8's lb.
    """
    n = keys.size
    ln = e
    lc_eff = min(lc, e - 1)
    lb = max(1, e - lc_eff)
    ln_minus_lb = ln - lb
    n_bins = 1 << ln_minus_lb
    bin_shift = np.uint32(p - ln_minus_lb)
    entry_bits = lb + (p - ln)  # = p - lc_eff

    t0 = time.perf_counter()
    bin_starts = precount_and_starts(keys, bin_shift, n_bins)
    hist = np.zeros(n_bins, dtype=np.int64)
    sbatch_mem = np.empty(n, dtype=np.uint16)
    bin_counts = np.zeros(n_bins, dtype=np.int32)
    bin_wp = bin_starts.copy()
    frmw_io_fast_process_u16(keys, hist, ln_minus_lb, lb, p, ln,
                             sbatch_mem, bin_starts, bin_counts, lut, bin_wp)
    #fsf_binsort_hist_only(sbatch_mem, bin_starts, bin_counts,
    #                      n_bins, entry_bits)
    fsf_binsort_reconstruct_u32(sbatch_mem, bin_starts, bin_counts,
                          n_bins, entry_bits)
    return time.perf_counter() - t0


def run_fsf_dram(keys, e, lut, lc=16):
    """FSF with DRAM-stride scatter + DRAM-stride binsort.

    Defaults: lc=16, entry_bits = p - lc = 16, lb = e - lc.
    ldram = 32 (512//16).
    """
    n = keys.size
    ln = e
    lc_eff = min(lc, e - 1)
    lb = max(1, e - lc_eff)
    ln_minus_lb = ln - lb
    n_bins = 1 << ln_minus_lb
    bin_shift = np.uint32(p - ln_minus_lb)
    entry_bits = lb + (p - ln)
    ldram = 512 // 16
    nbins = n_bins

    t0 = time.perf_counter()
    bin_starts = precount_and_starts(keys, bin_shift, n_bins)
    hist = np.zeros(n_bins, dtype=np.int64)
    sbatch_mem = np.empty(n, dtype=np.uint16)
    bin_counts = np.zeros(n_bins, dtype=np.int32)
    bin_wp = np.zeros(n_bins, dtype=np.int64)
    frmw_io_fast_process_u16_dram(keys, hist, ln_minus_lb, lb, p, ln,
                                   sbatch_mem, bin_starts, bin_counts, lut, bin_wp)
    fsf_binsort_dram(sbatch_mem, bin_counts, n_bins, entry_bits,
                     ldram, nbins)
    return time.perf_counter() - t0


def run_fsfs(keys, e, lut, lc=16):
    """FSFS: scatter (uint16) + 4-bit packed histogram bin sort.

    Same as FSF but scatter writes uint16 entries instead of uint32.
    Defaults: lc=16, entry_bits = p - lc = 16, lb = e - lc.
    """
    n = keys.size
    ln = e
    lc_eff = min(lc, e - 1)
    lb = max(1, e - lc_eff)
    ln_minus_lb = ln - lb
    n_bins = 1 << ln_minus_lb
    bin_shift = np.uint32(p - ln_minus_lb)
    entry_bits = lb + (p - ln)

    t0 = time.perf_counter()
    bin_starts = precount_and_starts(keys, bin_shift, n_bins)
    hist = np.zeros(n_bins, dtype=np.int64)
    sbatch_mem = np.empty(n, dtype=np.uint16)
    bin_counts = np.zeros(n_bins, dtype=np.int32)
    bin_wp = bin_starts.copy()
    frmw_io_fast_process_u16(keys, hist, ln_minus_lb, lb, p, ln,
                             sbatch_mem, bin_starts, bin_counts, lut, bin_wp)
    return time.perf_counter() - t0


def bench_all(keys, e, lb, lut):
    """Run kdb+, FS8, FSC8, FSF once each. Returns (t_kdb, t_fs8, t_fsc8, t_fsf)."""
    n = keys.size
    ln = e
    ln_minus_lb = ln - lb
    n_bins = 1 << ln_minus_lb
    bin_shift = np.uint32(p - ln_minus_lb)

    # --- kdb+ ---
    t0 = time.perf_counter()
    kdb_radix_sort(keys)
    t_kdb = time.perf_counter() - t0

    # --- FS8: scatter + bin sort ---
    t0 = time.perf_counter()
    bin_starts = precount_and_starts(keys, bin_shift, n_bins)
    hist = np.zeros(n_bins, dtype=np.int64)
    sbatch_mem = np.empty(n, dtype=np.uint32)
    bin_counts = np.zeros(n_bins, dtype=np.int32)
    bin_wp = bin_starts.copy()
    frmw_io_fast_process(keys, hist, ln_minus_lb, lb, p, ln,
                         sbatch_mem, bin_starts, bin_counts, lut, bin_wp)
    countsort_bins_8bit(sbatch_mem, bin_starts, bin_counts,
                        n_bins, lb, ln_minus_lb, p, ln)
    t_fs8 = time.perf_counter() - t0

    # --- FSC8: scatter + bin sort + reconstruct ---
    t0 = time.perf_counter()
    bin_starts2 = precount_and_starts(keys, bin_shift, n_bins)
    hist2 = np.zeros(n_bins, dtype=np.int64)
    sbatch_mem2 = np.empty(n, dtype=np.uint32)
    bin_counts2 = np.zeros(n_bins, dtype=np.int32)
    bin_wp2 = bin_starts2.copy()
    frmw_io_fast_process(keys, hist2, ln_minus_lb, lb, p, ln,
                         sbatch_mem2, bin_starts2, bin_counts2, lut, bin_wp2)
    countsort_bins_8bit(sbatch_mem2, bin_starts2, bin_counts2,
                        n_bins, lb, ln_minus_lb, p, ln)
    output = np.empty(n, dtype=np.uint32)
    reconstruct_keys(sbatch_mem2, bin_starts2, bin_counts2,
                     n_bins, ln_minus_lb, p, output)
    t_fsc8 = time.perf_counter() - t0

    # --- FSF: uses its own encapsulated defaults ---
    t_fsf = run_fsf(keys, e, lut)

    return t_kdb, t_fs8, t_fsc8, t_fsf


def main():
    lut = _build_reverse_lut()

    # JIT warmup
    print("JIT warmup...")
    k_warm = np.random.randint(0, 2**32, size=1 << 14, dtype=np.uint32)
    kdb_radix_sort(k_warm)
    bench_all(k_warm, 14, max(1, 14 - 8), lut)
    print("Done.\n")

    print(f"{'e':>4} {'n':>12} {'lb':>3} {'bins':>6} {'entry':>5} | "
          f"{'kdb+':>10} {'FS8':>10} {'FSC8':>10} {'FSF':>10} | "
          f"{'FS8/kdb':>7} {'FSC8/kdb':>8} {'FSF/kdb':>7}")
    print("-" * 100)

    for e in [18, 20, 22, 24, 26, 28]:
        n = 1 << e
        lb = e - 8 if e < 22 else e - 10
        lb = max(1, lb)
        ln_minus_lb = e - lb
        n_bins = 1 << ln_minus_lb
        entry_bits = lb + (p - e)

        np.random.seed(42)
        keys = np.random.randint(0, 1 << p, size=n, dtype=np.uint32)

        t_kdb, t_fs8, t_fsc8, t_fsf = bench_all(keys, e, lb, lut)

        kdb_mks = n / t_kdb / 1e6
        fs8_mks = n / t_fs8 / 1e6
        fsc8_mks = n / t_fsc8 / 1e6
        fsf_mks = n / t_fsf / 1e6

        print(f"{e:>4} {n:>12,} {lb:>3} {n_bins:>6} {entry_bits:>5} | "
              f"{kdb_mks:>8.1f}M/s {fs8_mks:>8.1f}M/s {fsc8_mks:>8.1f}M/s {fsf_mks:>8.1f}M/s | "
              f"{fs8_mks/kdb_mks:>5.2f}x {fsc8_mks/kdb_mks:>6.2f}x {fsf_mks/kdb_mks:>5.2f}x")

    print()
    print("FS8  = scatter + 8-bit radix bin sort")
    print("FSC8 = scatter + 8-bit radix bin sort + reconstruct")
    print("FSF  = scatter + 4-bit packed histogram bin sort (lc=16, entry_bits=16)")
    print("kdb+ = LSB radix sort (4-pass, 32-bit)")
    print()
    print("FS8/FSC8: lb = e-8 (e<22), e-10 (e>=22).")
    print("FSF:      lc=16, lb = e-16, entry_bits = 16 (defaults in run_fsf).")
    print(f"p={p}.")
    print(f"Platform: {sys.platform}, Python {sys.version.split()[0]}")
    import numba
    print(f"Numba {numba.__version__}, NumPy {np.__version__}")


if __name__ == "__main__":
    main()
