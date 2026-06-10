"""Test fractalsort_cpu: correctness of sort, get_item, and reconstruction."""
import numpy as np
import sys
import time

from fractalsort_core import fractalsort

e = int(sys.argv[1]) if len(sys.argv) > 1 else 18
lb_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
p = 32
n = 1 << e

print(f'=== FractalSort Correctness Test ===')
print(f'n = 2^{e} = {n:,}, p = {p}')

np.random.seed(42)
keys = np.random.randint(0, 1 << p, size=n, dtype=np.uint32)

# Sort
print('\nRunning fractalsort...', flush=True)
t0 = time.perf_counter()
result = fractalsort(keys, p=p, lb=lb_arg)
dt = time.perf_counter() - t0
print(f'Done in {dt*1000:.0f} ms ({n/dt/1e6:.1f} Mkeys/s)')
print(f'  ln={result.ln}, lb={result.lb}, bins={result.n_bins}, '
      f'entry_bits={result.entry_bits}, n={len(result):,}')

all_pass = True

# Test 1: Reconstruct all keys — verify same multiset as input
print('\nTest 1: Full reconstruction (multiset check)...', flush=True)
recon = result.reconstruct_all()
orig_sorted = np.sort(keys)
recon_sorted = np.sort(recon)
if np.array_equal(orig_sorted, recon_sorted):
    print('  PASS: all keys recovered (multiset match)')
else:
    n_match = np.sum(orig_sorted == recon_sorted)
    print(f'  FAIL: {n_match}/{n} match after sorting')
    diff_idx = np.where(orig_sorted != recon_sorted)[0][:5]
    for di in diff_idx:
        print(f'    pos {di}: expected 0x{orig_sorted[di]:08x}, got 0x{recon_sorted[di]:08x}')
    all_pass = False

# Test 2: get_item consistent with reconstruct_all
print('\nTest 2: get_item vs reconstruct_all...', flush=True)
test_positions = [0, 1, n // 4, n // 2, 3 * n // 4, n - 2, n - 1]
ok = True
for pos in test_positions:
    got = result.get_item(pos)
    expected = recon[pos]
    if got != expected:
        print(f'  FAIL at pos {pos}: recon=0x{expected:08x}, get_item=0x{got:08x}')
        ok = False
if ok:
    print(f'  PASS: all {len(test_positions)} positions match reconstruct_all')
else:
    all_pass = False

# Test 3: get_item random positions
print('\nTest 3: get_item random positions vs reconstruct_all...', flush=True)
n_random = min(1000, n)
rng = np.random.RandomState(123)
random_positions = rng.randint(0, n, size=n_random)
n_ok = 0
for pos in random_positions:
    got = result.get_item(int(pos))
    if got == recon[pos]:
        n_ok += 1
if n_ok == n_random:
    print(f'  PASS: all {n_random} random positions correct')
else:
    print(f'  FAIL: {n_ok}/{n_random} correct')
    all_pass = False

# Test 4: slice access
print('\nTest 4: slice access...', flush=True)
first_10 = result[0:10]
if np.array_equal(first_10, recon[:10]):
    print('  PASS: result[0:10] matches')
else:
    print('  FAIL: result[0:10] mismatch')
    all_pass = False

last_10 = result[-10:]
if np.array_equal(last_10, recon[-10:]):
    print('  PASS: result[-10:] matches')
else:
    print('  FAIL: result[-10:] mismatch')
    all_pass = False

# Test 5: tree-walk order is monotonic within bins
print('\nTest 5: entries sorted within each bin...', flush=True)
bin_ok = True
offset = 0
for b in range(result.n_bins):
    count = int(result.bin_counts[b])
    if count <= 1:
        offset += count
        continue
    # Check entries in sorted order are non-decreasing
    prev = result.sbatch_mem[offset + int(result.index_array[offset])]
    for sp in range(1, count):
        arrival = int(result.index_array[offset + sp])
        cur = result.sbatch_mem[offset + arrival]
        if cur < prev:
            print(f'  FAIL: bin {b}, sorted_pos {sp}: entry {cur} < prev {prev}')
            bin_ok = False
            break
        prev = cur
    offset += count
if bin_ok:
    print(f'  PASS: all {result.n_bins} bins have sorted entries')
else:
    all_pass = False

# Test 6: get_item latency
print('\nTest 6: get_item latency...', flush=True)
n_lookups = 10000
positions = rng.randint(0, n, size=n_lookups)
for i in range(100):
    _ = result.get_item(int(positions[i]))
t0 = time.perf_counter()
for i in range(n_lookups):
    _ = result.get_item(int(positions[i]))
dt = time.perf_counter() - t0
print(f'  {n_lookups} lookups in {dt*1000:.1f} ms ({dt/n_lookups*1e6:.2f} us/lookup)')

# Memory
entry_bytes = n * 4
index_bytes = n * 4
hist_bytes = result.n_bins * 8
total_mem = entry_bytes + index_bytes + hist_bytes
key_bytes = n * 4
trailing_bits = p - result.ln

print(f'\n=== Memory Analysis ===')
print(f'Input keys:     {key_bytes:>12,} bytes ({key_bytes/n:.1f} B/key)')
print(f'Entries:        {entry_bytes:>12,} bytes ({entry_bytes/n:.1f} B/key) '
      f'[{result.entry_bits} bits, stored uint32]')
print(f'Index array:    {index_bytes:>12,} bytes ({index_bytes/n:.1f} B/key) '
      f'[{result.lb} bits, stored int32]')
print(f'Histogram:      {hist_bytes:>12,} bytes')
print(f'TOTAL stored:   {total_mem:>12,} bytes ({total_mem/n:.1f} B/key)')

import math
packed_trailing = math.ceil(n * trailing_bits / 8)
packed_index = math.ceil(n * result.lb / 8)
packed_total = packed_trailing + packed_index + hist_bytes
print(f'\nPacked (trailing-only + lb-bit index):')
print(f'  Trailing:     {packed_trailing:>12,} bytes ({trailing_bits} bits/key)')
print(f'  Index:        {packed_index:>12,} bytes ({result.lb} bits/key)')
print(f'  Histogram:    {hist_bytes:>12,} bytes')
print(f'  TOTAL:        {packed_total:>12,} bytes ({packed_total/n:.2f} B/key)')

if all_pass:
    print(f'\n=== ALL TESTS PASSED ===')
else:
    print(f'\n=== SOME TESTS FAILED ===')
    sys.exit(1)
