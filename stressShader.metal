#include <metal_stdlib>
using namespace metal;

// Must exactly match the host code.
constant uint SPLIT_THREADS = 2;
constant uint FLAG_NOT_READY = 0;
constant uint FLAG_READY = 0x40000000;
constant uint FLAG_INCLUSIVE = 0x80000000;
constant uint FLAG_MASK = 0xC0000000;
constant uint VALUE_MASK = 0xffff;
constant uint SPLIT_READY = 3;

constant uint BLOCK_DIM = 32;
constant uint ERROR_TYPE_MESSAGE = 1u;
constant uint ERROR_TYPE_SHUFFLE_READY = 2u;
constant uint ERROR_TYPE_SHUFFLE_INC = 3u;
constant uint ERROR_TYPE_SGSIZE  = 4u;

uint ballot(bool pred) {
  return as_type<uint2>((simd_vote::vote_t)simd_ballot(pred)).x;
}

uint join(uint mine, uint tid) {
  const uint xord = tid ^ 1;
  const uint theirs = simd_shuffle(mine, xord);
  return mine << 16 * tid | theirs << 16 * xord;
}

uint split(uint x, uint tid) {
  return x >> tid * 16 & VALUE_MASK;
}

typedef uint2 errType[2];
bool messagePassingCheck(uint tid, uint flag_payload, uint lookback_id,
                         uint tile_id, device errType* errors) {
     bool is_valid_payload =
                        (flag_payload == FLAG_NOT_READY ||
                         flag_payload == (split(1024, tid) | FLAG_READY) ||
                         flag_payload == (split((lookback_id + 1) * 1024, tid) | FLAG_INCLUSIVE));
    if (!is_valid_payload) {
        errors[tile_id][tid].x = ERROR_TYPE_MESSAGE;
        errors[tile_id][tid].y = flag_payload;
        return true;
    }
    return false;
}

bool shuffleCheckReady(uint tid, uint prev_red, uint lookback_id,
                       uint tile_id, device errType* errors) {
    uint expected_value = (tile_id - lookback_id) * 1024;
    if (prev_red != expected_value) {
        errors[tile_id][tid].x = ERROR_TYPE_SHUFFLE_READY;
        errors[tile_id][tid].y = prev_red;
        return true;
    }
    return false;
}

bool shuffleCheckInclusive(uint tid, uint prev_red, uint lookback_id,
                           uint tile_id, device errType* errors) {
    uint expected_value = tile_id * 1024;
    if (prev_red != expected_value) {
        errors[tile_id][tid].x = ERROR_TYPE_SHUFFLE_INC;
        errors[tile_id][tid].y = prev_red;
        return true;
    }
    return false;
}

typedef atomic_uint splitType[2];
kernel void stress(uint3 threadid [[thread_position_in_threadgroup]],
                    uint laneid [[thread_index_in_simdgroup]],
                    uint sgSize [[threads_per_simdgroup]],
                    device atomic_uint* scan_bump [[buffer(0)]],
                    device splitType* scan [[buffer(1)]],
                    device errType* errors [[buffer(2)]]) {
  // BLOCK_DIM must equal subgroup_size
  if (BLOCK_DIM != sgSize) {
    errors[0][0].x = ERROR_TYPE_SGSIZE;
    return;
  }

  const bool is_split_thread = threadid.x < SPLIT_THREADS;
  uint tile_id = 0;
  if (threadid.x == 0) {
    tile_id = atomic_fetch_add_explicit(&scan_bump[0], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tile_id = simd_broadcast(tile_id, 0);

  if (is_split_thread) {
    const uint t = split(1024, threadid.x) | (tile_id == 0 ? FLAG_INCLUSIVE : FLAG_READY);
    atomic_store_explicit(&scan[tile_id][threadid.x], t, memory_order_relaxed);
  }

  if (tile_id != 0) {
    uint prev_red = 0;
    uint lookback_id = tile_id - 1;
    bool errEncountered = false;

    while (true) {
      uint flag_payload = is_split_thread ?
                        atomic_load_explicit(&scan[lookback_id][threadid.x], memory_order_relaxed) :
                        0;
      if (!errEncountered && is_split_thread) {
        errEncountered = messagePassingCheck(threadid.x, flag_payload, lookback_id,
                                             tile_id, errors);
      }

      if (ballot((flag_payload & FLAG_MASK) > FLAG_NOT_READY) == SPLIT_READY) {
        uint inc_bal = ballot((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE);
        if (inc_bal != 0) {
          while(inc_bal != SPLIT_READY) {
            flag_payload = is_split_thread ?
                        atomic_load_explicit(&scan[lookback_id][threadid.x], memory_order_relaxed) :
                        0;
            inc_bal = ballot((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE);
          }
          if (!errEncountered && is_split_thread) {
            errEncountered = messagePassingCheck(threadid.x, flag_payload, lookback_id,
                                                 tile_id, errors);
          }

          prev_red += join(flag_payload & VALUE_MASK, threadid.x);
          if (!errEncountered && is_split_thread) {
            errEncountered = shuffleCheckInclusive(threadid.x, prev_red, lookback_id,
                                                   tile_id, errors);
          }

          if (is_split_thread) {
            const uint t = split(prev_red + 1024, threadid.x) | FLAG_INCLUSIVE;
            atomic_store_explicit(&scan[tile_id][threadid.x], t, memory_order_relaxed);
          }
          break;
        } else {
          prev_red += join(flag_payload & VALUE_MASK, threadid.x);
          if (!errEncountered && is_split_thread) {
            errEncountered = shuffleCheckReady(threadid.x, prev_red, lookback_id,
                                               tile_id, errors);
          }
          lookback_id -= 1;
        }
      } // else load fresh values.
    }
  }
}