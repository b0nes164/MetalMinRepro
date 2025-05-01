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

// We choose a workgroup dimension with the exact size of an Apple subgroup (typically 32)
// to ensure subgroup operations behave as expected.
constant uint BLOCK_DIM = 32;
constant uint ERROR_TYPE_MESSAGE = 1u;
constant uint ERROR_TYPE_SHUFFLE_READY = 2u;
constant uint ERROR_TYPE_SHUFFLE_INC = 3u;
constant uint ERROR_TYPE_SGSIZE = 4u;

// Get the ballot back as a uint. Lop off the upper bits, as we require a 32 simdgroup size, and
// will never need them. WGSL equivalent: subgroupBallot(pred).x
uint ballot(bool pred) { return as_type<uint2>((simd_vote::vote_t)simd_ballot(pred)).x; }

// Once the flags of both split threads match, "join" the values together by a one-step butterfly
// shuffle followed by bit manipulation. Post-join, both split threads MUST have the same result.
uint join(uint mine, uint tid) {
    const uint xord = tid ^ 1;
    const uint theirs = simd_shuffle(mine, xord);  // WGSL: unsafeShuffle(mine, xor)
    return mine << 16 * tid | theirs << 16 * xord;
}

// Prior to storing the values in global memory, split the value into its constituent 16-bit parts.
uint split(uint x, uint tid) { return x >> tid * 16 & VALUE_MASK; }

// The error buffer is made up of array<uint2, 2>. Each thread of every tile may post an error code
// and the incorrect value it found. Because one downstream incorrect results in errors in all
// upstream results, we are primarily interested in the FIRST incorrect error code.
typedef uint2 errType[2];

// Checks the flag_payload loaded from global memory after every load.
// Because the inputs are constant, each tile has only 3 valid values:
bool messagePassingCheck(uint tid, uint flag_payload, uint lookback_id, uint tile_id,
                         device errType* errors) {
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

// Checks the post-joined value in the "both flags ready" branch. If a valid data was passed between
// workgroups, then the post shuffle value must exactly match the expected sum. This branch is taken
// when both split threads signal READY (but not INCLUSIVE). This branch performs less atomic
// operations, so it may be less liable to encounter errors.
bool shuffleCheckReady(uint tid, uint prev_red, uint lookback_id, uint tile_id,
                       device errType* errors) {
    uint expected_value = (tile_id - lookback_id) * 1024;
    if (prev_red != expected_value) {
        errors[tile_id][tid].x = ERROR_TYPE_SHUFFLE_READY;
        errors[tile_id][tid].y = prev_red;
        return true;
    }
    return false;
}

// Checks the post-joined value in the "both flags inclusive" branch. If a valid data was passed
// (especially after waiting for matching INCLUSIVE flags), then the post shuffle value must exactly
// match the expected sum. This branch is taken when both split threads have signaled INCLUSIVE.
// When a single inclusive flag is encountered initially, threads must wait until the matching pair
// is also INCLUSIVE, resulting in increased atomic operations, and more opportunities for issues
// before this check.
bool shuffleCheckInclusive(uint tid, uint prev_red, uint lookback_id, uint tile_id,
                           device errType* errors) {
    uint expected_value = tile_id * 1024;
    if (prev_red != expected_value) {
        errors[tile_id][tid].x = ERROR_TYPE_SHUFFLE_INC;
        errors[tile_id][tid].y = prev_red;
        return true;
    }
    return false;
}

// This kernel runs the inter-workgroup portion of a Chained-Scan with Lookback. It does not include
// any fallback routine, so running it on devices without FPG may result in unexpected behavior. In
// our case we use this scenario to check for simdgroup divergence or message passing issues.
typedef atomic_uint splitType[2];
kernel void stress(uint3 threadid [[thread_position_in_threadgroup]],
                   uint laneid [[thread_index_in_simdgroup]],
                   uint sgSize [[threads_per_simdgroup]],
                   device atomic_uint* scan_bump [[buffer(0)]],
                   device splitType* scan [[buffer(1)]],
                   device errType* errors [[buffer(2)]]) {
    if (BLOCK_DIM != sgSize) {
        errors[0][0].x = ERROR_TYPE_SGSIZE;
        return;
    }
    const bool is_split_thread = threadid.x < SPLIT_THREADS;

    // Acquire partition index by atomically bumping global memory. This guarantees that predecessor
    // workgroups are either completed or at least resident on an SM.
    uint tile_id = 0;
    if (threadid.x == 0) {
        tile_id = atomic_fetch_add_explicit(&scan_bump[0], 1u, memory_order_relaxed);
    }
    // Safety barrier, don't want possible divergence here before broadcast.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tile_id = simd_broadcast(tile_id, 0);

    // The split threads post the values into global memory. The first tile posts FLAG_INCLUSIVE
    // because it has no predecessor tiles, so it already contains the inclusive reduction of
    // preceeding tiles.
    //
    // The value to post, 1024u32, is split into its constituent upper and lower 16 bits. Each split
    // thread places its portion of the bits into global memory. The value posted by each thread
    // is bitpacked to contain a copy of the FLAG.
    //
    // The layout of the scan buffer looks like this:
    // scan[tile_id][split_thread_index]
    // Each entry is a u32: (16-bit split value | Flags using upper bits)
    // Example for initial posting (value = 1024u):
    //
    //   scan_buffer_address   <---------------- Tile ID ----------------->
    //   |                     Tile 0            Tile 1                      Tile N (example)
    //   V Index within Tile   +----------------+---------------------+ ... +---------------------------+
    //   [0] (threadid.x=0)  | LSBs | FLAG_INCL| | LSBs | FLAG_READY |       | LSBs | FLAG_READY   |
    //                         +---------------------------+----------+     +---------------------------+
    //   [1] (threadid.x=1)  | MSBs | FLAG_INCL| | MSBs | FLAG_READY |       | MSBs | FLAG_READY   |
    //                         +----------------+---------------------+ ... +---------------------------+
    // LSBs/MSBs refer to the lower/upper 16 bits of the value being posted by the tile (initially
    // 1024u). Initially, only Tile 0 gets FLAG_INCLUSIVE. Other tiles get FLAG_READY. These flags
    // can be updated to FLAG_INCLUSIVE later during the lookback phase.
    //
    if (is_split_thread) {
        const uint t = split(1024, threadid.x) | (tile_id == 0 ? FLAG_INCLUSIVE : FLAG_READY);
        atomic_store_explicit(&scan[tile_id][threadid.x], t, memory_order_relaxed);
    }

    // The goal of lookback is for each workgroup to traverse backwards along the scan buffer,
    // calculating reduction of previous tiles as it goes. If a tile is encountered with
    // FLAG_INCLUSIVE, the traversal can exit early because we guarantee that a tile with
    // FLAG_INCLUSIVE contains the inclusive reduction of all tiles up to itself.
    //
    // To enable this early exiting, whenever a tile completes its traversal (finds an INCLUSIVE
    // predecessor or reaches tile 0), it updates its own posting in the scan buffer.
    // It overwrites its previous FLAG_READY value with split(prev_red + 1024u) | FLAG_INCLUSIVE.
    // Here, (prev_red + 1024u) is the inclusive sum for the current tile.
    //
    // Because tile updates are made across separate threads (the SPLIT_THREADS) and values,
    // the split threads must coordinate with each other using subgroup operations (ballot, shuffle)
    // to ensure that the FLAGS of the data they read and write are in an exactly matching state
    // before combining values or making decisions.
    //
    // Lookback is a one-way producer-consumer algorithm. It requires forward progress guarantees
    // (FPG) for the in-flight workgroup with the lowest tile_id. Attempting to run this algorithm
    // without FPG (or on hardware that cannot guarantee it sufficiently for all active workgroups)
    // can result in prolonged spinning (especially in the while loops waiting for flags) and
    // increased stress on the memory system.

    // The first workgroup (tile_id == 0), already has posted its FLAG_INCLUSIVE, so it skips this
    // lookback operation.
    if (tile_id != 0) {
        // This holds the reduction of the previous tiles. Each split thread maintains its own copy
        // of this accumulating sum. Note value is not "split" initially---it is the full u32 sum.
        // Modifications to it must operate on the full u32 value. Thus, prior to adding a
        // value from a predecessor tile (which is stored split), we must "join" the split parts.
        uint prev_red = 0;

        // Each workgroup begins its traversal with its immediate predecessor tile.
        uint lookback_id = tile_id - 1;
        bool errEncountered = false;  // Per-thread error flag for the current workgroup

        while (true) {
            // The split threads load their respective packed value in from global memory
            // (scan[lookback_id]). Non-split threads get a 0, as they don't participate in
            // loading/processing this data.
            uint flag_payload =
                is_split_thread
                    ? atomic_load_explicit(&scan[lookback_id][threadid.x], memory_order_relaxed)
                    : 0;

            if (!errEncountered && is_split_thread) {
                errEncountered =
                    messagePassingCheck(threadid.x, flag_payload, lookback_id, tile_id, errors);
            }

            // Next, the split threads check (via ballot) if both threads loaded a flag indicating
            // data is READY or INCLUSIVE. SPLIT_READY (3, which is 0b11) means both of the first
            // two threads in the ballot (our split threads) voted true.
            if (ballot((flag_payload & FLAG_MASK) > FLAG_NOT_READY) == SPLIT_READY) {
                // Both split threads have found data that is at least READY.
                // Now, check if an INCLUSIVE flag was loaded by any of the split threads.
                uint inc_bal = ballot((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE);

                // Because states change in a strict order NOT_READY -> READY -> INCLUSIVE and never
                // revert, once an INCLUSIVE is read by one split thread, we have no other choice
                // but to wait until its matching pair (the other split thread) also reads an
                // INCLUSIVE state from its part of the scan entry.
                if (inc_bal != 0) {  // At least one split thread read INCLUSIVE.
                    while (inc_bal !=
                           SPLIT_READY) {  // Wait until *both* split threads read INCLUSIVE.
                        // Spin-load until the condition is met.
                        flag_payload = is_split_thread
                                           ? atomic_load_explicit(&scan[lookback_id][threadid.x],
                                                                  memory_order_relaxed)
                                           : 0;
                        inc_bal = ballot((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE);
                    }

                    // Both threads have now loaded INCLUSIVE from scan[lookback_id].
                    if (!errEncountered && is_split_thread) {
                        errEncountered = messagePassingCheck(threadid.x, flag_payload, lookback_id,
                                                             tile_id, errors);
                    }

                    // Once both threads have loaded INCLUSIVE, rejoin the value parts. Each split
                    // thread calculates the joined value from its part and the other's part.
                    // (flag_payload & VALUE_MASK) extracts the 16-bit data part.
                    prev_red += join(flag_payload & VALUE_MASK, threadid.x);
                    if (!errEncountered && is_split_thread) {
                        errEncountered = shuffleCheckInclusive(threadid.x, prev_red, lookback_id,
                                                               tile_id, errors);
                    }

                    // The lookback has found an inclusive sum. This 'prev_red' is the sum of all
                    // tiles *before* the current one. Add this tile's own contribution (1024) to
                    // 'prev_red' to get the inclusive sum *for this tile*. Then, split this new
                    // inclusive sum, pack it with FLAG_INCLUSIVE, and post to global memory for
                    // this tile.
                    if (is_split_thread) {
                        const uint t = split(prev_red + 1024, threadid.x) | FLAG_INCLUSIVE;
                        atomic_store_explicit(&scan[tile_id][threadid.x], t, memory_order_relaxed);
                    }

                    // The lookback is complete for this workgroup, exit the while loop.
                    break;
                } else {
                    // Both threads loaded flags greater than NOT_READY, but neither were INCLUSIVE.
                    // This means both threads must have loaded READY.
                    // Join the value from scan[lookback_id] and add it to the reduction 'prev_red'.
                    prev_red += join(flag_payload & VALUE_MASK, threadid.x);
                    if (!errEncountered && is_split_thread) {
                        errEncountered =
                            shuffleCheckReady(threadid.x, prev_red, lookback_id, tile_id, errors);
                    }
                    lookback_id -= 1;
                }  // else, ballot condition not met (SPLIT_READY), means at least one split thread
                   // read NOT_READY.
            }
        }
        // Implicit barrier at end of kernel execution for the workgroup.
    }
}