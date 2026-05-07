"""frmw_io: Pure parallel FRMW with streaming sbatch buffering.

Architecture:
1. Counter tree levels 0..lc-1 in cache (half cache budget).
2. Route each key to one of 2^(ln-lb) bins (tree-ordered: MSBs of bin_id = bits 0..lc-1).
   Store entries in sbatch array (within-bin sort key: offset bits + trailing bits).
3. Buffer entries in cache; flush to memory when full.
4. Final pass: sort within each bin by entry value, produce lb-bit index array.

Bit decomposition of a p-bit key:
  bits 0..ln-lb-1    -> bin_id (tree-ordered, ln-lb bits). MSBs = bits 0..lc-1.
  bits ln-lb..ln-1   -> within-bin offset (lb bits, recovered by step 4 sort)
  bits ln..p-1       -> trailing (p-ln bits, stored in entry)

Sbatch entry stores: within-bin sort key = bits ln-lb..ln-1 + bits ln..p-1 = lb + (p-ln) bits.
  These are the bits NOT encoded by the bin_id.
  At p=32, ln=30, lb=10: entry = 10 + 2 = 12 bits (stored as uint16).

The routing bits (bits lc..ln-lb-1) are NOT stored in entries — they're
redundant with the bin_id LSBs.

Bin_id computation (tree-ordered):
  bin_id = bit_reverse(key & ((1<<(ln-lb))-1), ln-lb)
  This places bit 0 of key as MSB of bin_id.
"""
import numpy as np
from numba import njit


@njit(nogil=True, fastmath=True, cache=True)
def _bit_reverse(val, nbits):
    """Reverse the lowest nbits of val."""
    result = np.int64(0)
    for i in range(nbits):
        result = (result << 1) | (val & 1)
        val >>= 1
    return result


@njit(nogil=True, fastmath=True, cache=True)
def frmw_io_process(keys, cnts, widths, per_word, word_offsets, lc, ln, lb, p,
                    sbatch_mem, sbatch_bin_counts, mem_write_pos):
    """Process one batch of keys through frmw_io.

    Sbatch entry = offset_bits (lb) | trailing_bits (p-ln), stored as uint16 or uint32.
    Total entry bits = lb + (p-ln).
    """
    nk = keys.size
    ln_minus_lb = ln - lb
    n_bins = np.int64(1) << ln_minus_lb
    offset_bits = lb          # bits ln-lb..ln-1 of key
    trailing_bits = p - ln    # bits ln..p-1 of key
    entry_bits = offset_bits + trailing_bits

    # For tree-ordered sort within bin: store bits ln-lb..ln-1 in tree order (bit-reversed)
    # plus trailing bits. Entry = (tree_reversed_offset << trailing_bits) | trailing
    offset_mask = np.uint32((1 << lb) - 1)
    trailing_mask = np.uint32((1 << trailing_bits) - 1)

    # Cache buffer: count-sort entire batch by bin_id before writing to memory
    bin_counts = np.zeros(n_bins, dtype=np.int32)
    entries = np.empty(nk, dtype=np.uint32)
    bin_ids = np.empty(nk, dtype=np.int64)

    for i in range(nk):
        ki = keys[i]
        ki64 = np.uint64(ki)

        # Counter updates for levels 0..lc-1
        bin_idx = np.int64(0)
        for l in range(lc):
            w = widths[l]
            pw = per_word[l]
            word_idx = word_offsets[l] + bin_idx // pw
            bit_pos = np.uint64((bin_idx % pw) * w)
            cnts[np.intp(word_idx)] += np.uint64(1) << bit_pos
            bin_idx = bin_idx | (np.int64((ki64 >> np.uint64(l)) & np.uint64(1)) << l)

        # Compute bin_id (tree-ordered): bit_reverse of lower (ln-lb) key bits
        bid = _bit_reverse(np.int64(ki) & np.int64((1 << ln_minus_lb) - 1), ln_minus_lb)
        bin_ids[i] = bid
        bin_counts[bid] += 1

        # Compute entry: tree-ordered offset (bits ln-lb..ln-1) | trailing (bits ln..p-1)
        # Offset in tree order = bit_reverse(bits ln-lb..ln-1, lb)
        raw_offset = (ki >> np.uint32(ln_minus_lb)) & offset_mask
        tree_offset = np.uint32(_bit_reverse(np.int64(raw_offset), lb))
        trailing = (ki >> np.uint32(ln)) & trailing_mask
        entry = (tree_offset << np.uint32(trailing_bits)) | trailing
        entries[i] = entry

    # Scatter entries directly to per-bin regions in sbatch_mem.
    # sbatch_bin_counts[b] is the current write offset within bin b's region.
    # bin_starts[b] must be pre-computed so that bin b's region starts at bin_starts[b].
    # We use mem_write_pos as per-bin write cursors (passed as bin_write_pos array).
    # NOTE: caller must pre-allocate sbatch_mem with per-bin regions using bin_starts.

    for i in range(nk):
        b = bin_ids[i]
        wp = mem_write_pos[b]
        sbatch_mem[wp] = entries[i]
        mem_write_pos[b] = wp + 1

    # Update per-bin counts
    for b in range(n_bins):
        sbatch_bin_counts[b] += bin_counts[b]


@njit(nogil=True, fastmath=True, cache=True)
def frmw_io_finalize(sbatch_mem, sbatch_bin_counts, n_bins, entry_bits):
    """Step 4: Sort within each bin by entry value, produce lb-bit index array.

    After sorting, position i in sorted order = tree-walk position i within the bin.
    The index array maps: sorted_position -> storage_position (arrival pos within bin).
    This lets us read entries in tree-sorted order.

    Uses insertion sort (bins are small, ~256 entries at target scale).
    """
    total = np.int64(0)
    for b in range(n_bins):
        total += np.int64(sbatch_bin_counts[b])

    # index_array[global_offset + sorted_pos] = arrival_position_within_bin
    index_array = np.empty(total, dtype=np.uint16)

    offset = np.int64(0)
    for b in range(n_bins):
        count = np.int64(sbatch_bin_counts[b])
        if count == 0:
            continue

        # Build (entry_value, arrival_pos) pairs and sort by entry_value (stable)
        # Using insertion sort for simplicity
        perm = np.empty(count, dtype=np.int32)
        for i in range(count):
            perm[i] = np.int32(i)

        # Insertion sort by entry value (stable)
        for i in range(1, count):
            key_val = sbatch_mem[offset + perm[i]]
            j = i - 1
            tmp = perm[i]
            while j >= 0 and sbatch_mem[offset + perm[j]] > key_val:
                perm[j + 1] = perm[j]
                j -= 1
            perm[j + 1] = tmp

        # Write: index_array[sorted_pos] = arrival_pos
        for i in range(count):
            index_array[offset + i] = np.uint16(perm[i])

        offset += count

    return index_array


@njit(nogil=True, fastmath=True, cache=True)
def frmw_io_reconstruct(sbatch_mem, index_array, sbatch_bin_counts,
                        n_bins, ln, lb, lc, p, ln_minus_lb):
    """Reconstruct all keys from sbatch data in tree-sorted order.

    Walks bins in bin_id order (= tree order for bits 0..ln-lb-1).
    Within each bin, reads entries in sorted order (= tree order for bits ln-lb..ln-1).
    Extracts trailing bits from entry.
    """
    total = np.int64(0)
    for b in range(n_bins):
        total += np.int64(sbatch_bin_counts[b])

    keys_out = np.empty(total, dtype=np.uint32)
    trailing_bits = p - ln
    trailing_mask = np.uint32((1 << trailing_bits) - 1)
    offset_mask = np.uint32((1 << lb) - 1)

    offset = np.int64(0)
    out_idx = np.int64(0)

    for b in range(n_bins):
        count = np.int64(sbatch_bin_counts[b])
        if count == 0:
            continue

        # Recover bits 0..ln-lb-1 from bin_id (reverse the tree ordering)
        key_low = np.uint32(0)
        for bit in range(ln_minus_lb):
            if (np.int64(b) >> np.int64(ln_minus_lb - 1 - bit)) & 1:
                key_low |= np.uint32(1) << np.uint32(bit)

        # Read entries in sorted order (tree order within bin)
        for sp in range(count):
            # index_array gives us: for sorted_pos sp, the arrival position
            arrival_pos = np.int64(index_array[offset + sp])
            entry = sbatch_mem[offset + arrival_pos]

            # Entry = (tree_offset << trailing_bits) | trailing
            trailing = entry & trailing_mask
            tree_offset = (entry >> np.uint32(trailing_bits)) & offset_mask

            # Recover bits ln-lb..ln-1 from tree_offset (reverse bit order)
            offset_bits_val = np.uint32(0)
            for bit in range(lb):
                if (tree_offset >> np.uint32(lb - 1 - bit)) & 1:
                    offset_bits_val |= np.uint32(1) << np.uint32(bit)

            # Full key = bits_0_to_ln-lb-1 | (bits_ln-lb_to_ln-1 << (ln-lb)) | (trailing << ln)
            key = key_low | (offset_bits_val << np.uint32(ln_minus_lb)) | (trailing << np.uint32(ln))
            keys_out[out_idx] = key
            out_idx += 1

        offset += count

    return keys_out
