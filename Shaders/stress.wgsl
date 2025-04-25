
enable subgroups;
struct ScanParameters
{
    size: u32,
    unused_0: u32,
    unused_1: u32,
    unused_2: u32,
};

@group(0) @binding(0)
var<uniform> params : ScanParameters;

@group(0) @binding(1)
var<storage, read_write> scan_bump: atomic<u32>;

@group(0) @binding(2)
var<storage, read_write> scan: array<array<atomic<u32>, 2>>;

@group(0) @binding(3)
var<storage, read_write> errors: array<array<array<u32, 2>, 2>>;

const SPLIT_THREADS = 2u;
const FLAG_NOT_READY = 0u;
const FLAG_READY = 0x40000000u;
const FLAG_INCLUSIVE = 0x80000000u;
const FLAG_MASK = 0xC0000000u;
const VALUE_MASK = 0xffffu;
const SPLIT_READY = 3u;

var<workgroup> wg_broadcast: u32;

@diagnostic(off, subgroup_uniformity)
fn unsafeShuffle(x: u32, source: u32) -> u32 {
    return subgroupShuffle(x, source);
}

//lop off of the upper ballot bits;
//we never need them across all subgroup sizes
@diagnostic(off, subgroup_uniformity)
fn unsafeBallot(pred: bool) -> u32 {
    return subgroupBallot(pred).x;
}

fn join(mine: u32, tid: u32) -> u32 {
    let xor = tid ^ 1;
    let theirs = unsafeShuffle(mine, xor);
    return (mine << (16u * tid)) | (theirs << (16u * xor));
}

fn split(x: u32, tid: u32) -> u32 {
    return (x >> (tid * 16u)) & VALUE_MASK;
}

const ERROR_TYPE_MESSAGE = 1u;
const ERROR_TYPE_SHUFFLE = 2u;

// Because the inputs are constant, each tile has only 3 valid values:
fn messagePassingCheck(tid: u32, flag_payload: u32, lookback_id: u32, tile_id: u32) -> bool {
    if(!(flag_payload == FLAG_NOT_READY ||
         flag_payload == (split(1024u, tid) | FLAG_READY) ||
         flag_payload == (split((lookback_id + 1) * 1024u, tid) | FLAG_INCLUSIVE))) {
        errors[tile_id][tid][0] = ERROR_TYPE_MESSAGE;
        errors[tile_id][tid][1] = flag_payload;
        return true;
    }
    return false;
}

// If a valid data was passed between workgroups, then the post shuffle value must exactly match.
fn shuffleCheckReady(tid: u32, prev_red: u32, lookback_id: u32, tile_id: u32) -> bool {
    if (prev_red != (tile_id - lookback_id) * 1024u) {
        // atomic add error
        errors[tile_id][tid][0] = ERROR_TYPE_SHUFFLE;
        errors[tile_id][tid][1] = prev_red;
        return true;
    }
    return false;
}

fn  shuffleCheckInclusive(tid: u32, prev_red: u32, lookback_id: u32, tile_id: u32) -> bool {
    if (prev_red != tile_id * 1024u) {
        errors[tile_id][tid][0] = ERROR_TYPE_SHUFFLE;
        errors[tile_id][tid][1] = prev_red;
        return true;
    }
    return false;
}

// We choose a workgroup dimension with the exact size of an Apple subgroup
const BLOCK_DIM = 32u;
@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32) {
    let is_split_thread = threadid.x < SPLIT_THREADS;

    // Acquire partition index by atomically bumping global memory. This guarantees that predecessor
    // workgroups are either completed or at least resident on an SM.
    if (threadid.x == 0u) {
        wg_broadcast = atomicAdd(&scan_bump, 1u);
    }

    // Once acquired, broadcast the index to the rest of the workgroup.
    let tile_id = workgroupUniformLoad(&wg_broadcast);

    // The split threads post the values into global memory. The first tile posts FLAG_INCLUSIVE
    // because it has no predecessor tiles, so it already contains the inclusive reduction of
    // preceeding tiles.
    //
    // The value to post, 1024u32, is split into its constituent upper and lower 16 bits. Each split
    // thread places its portion of the bits into global memory. The value posted by each thread
    // is bitpacked to contain a copy of the FLAG.
    //
    // The layout of the scan buffer looks like this:
    //
    if (is_split_thread) {
        let t = split(1024u, threadid.x) | select(FLAG_READY, FLAG_INCLUSIVE, tile_id == 0u);
        atomicStore(&scan[tile_id][threadid.x], t);
    }

    // The goal of lookback is for each workgroup to traverse backwards along the scan buffer,
    // calculating reduction of previous tiles as it goes. If a tile is encountered with
    // FLAG_INCLUSIVE, the traversal can exit early because we guarantee that a tile with
    // FLAG_INCLUSIVE contains the inclusive reduction.
    //
    // To enable this early exiting, whenever a tile completes its traversal, it updates its posting
    // by overwriting the previous value with split(prev_red + 1024u | FLAG_INCLUSIVE). prev_red +
    // 1024u is the inclusive reduction.
    //
    // Because tile updates are made across separate threads and values, the split threads must
    // coordinate with each other using subgroup operations to ensure that the FLAGS of the data are
    // in an exactly matching state.
    //
    // Lookback is a one-way producer-consumer algorithm. It requires forward progress for the
    // in-flight workgroup with the lowest tile_id. Attempting to run this algorithm without forward
    // progress results in prolonged spinning and stress on the memory system.

    // The first workgroup, already has posted its FLAG_INCLUSIVE, so it skips this operation.
    if (tile_id != 0u) {
        // This holds the reduction of the previous tiles. Each split thread has its own copy. Note
        // value is not "split"---it is the full u32 value, so modifications to it must also be full
        // u32. Thus, prior to modifying it, we must "join" the values from the split threads.
        var prev_red = 0u;

        // Each workgroup begins its traversal with its immediate predecessor tile.
        var lookback_id = tile_id - 1u;
        var errorEncountered = false;
        while (true) {

            // The split threads load their respective packed value in from global memory.
            var flag_payload =
                select(0u, atomicLoad(&scan[lookback_id][threadid.x]), is_split_thread);
            if (!errorEncountered && is_split_thread) {
                errorEncountered =
                    messagePassingCheck(threadid.x, flag_payload, lookback_id, tile_id);
            }

            // Next, the split threads check to see if both threads loaded a READY or INCLUSIVE
            if (unsafeBallot((flag_payload & FLAG_MASK) > FLAG_NOT_READY) == SPLIT_READY) {

                // Next, the split threads to check if an INCLUSIVE was loaded.
                var incl_bal = unsafeBallot((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE);

                // Because states change in a strict order NOT_READY -> READY -> INCLUSIVE and never
                // revert, once an INCLUSIVE is read, we have no other choice but to wait until its
                // matching pair is also read.
                if (incl_bal != 0u) {
                    while (incl_bal != SPLIT_READY){
                        flag_payload =
                            select(0u, atomicLoad(&scan[lookback_id][threadid.x]), is_split_thread);
                        incl_bal = unsafeBallot((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE);
                    }
                    if (!errorEncountered && is_split_thread) {
                        errorEncountered =
                            messagePassingCheck(threadid.x, flag_payload, lookback_id, tile_id);
                    }

                    // Once both threads have loaded INCLUSIVE, rejoin the value.
                    prev_red += join(flag_payload & VALUE_MASK, threadid.x);
                    if (!errorEncountered && is_split_thread) {
                        errorEncountered =
                            shuffleCheckInclusive(threadid.x, prev_red, lookback_id, tile_id);
                    }

                    // Add the post-joined result to the initial result to produce the inclusive
                    // reduction. Then, split the value, pack it, and post to global memory.
                    if(is_split_thread){
                        let t = split(prev_red + 1024u, threadid.x) | FLAG_INCLUSIVE;
                        atomicStore(&scan[tile_id][threadid.x], t);
                    }

                    // The lookback is complete, exit.
                    break;
                } else {
                    // If both threads loaded flags greater than NOT_READY, but neither were
                    // INCLUSIVE, then both threads must have loaded READY. Join the value and add
                    // it to the reduction.
                    prev_red += join(flag_payload & VALUE_MASK, threadid.x);
                    if (!errorEncountered && is_split_thread) {
                        errorEncountered =
                            shuffleCheckReady(threadid.x, prev_red, lookback_id, tile_id);
                    }

                    // Continue looking back.
                    lookback_id -= 1u;
                }
            } // else load fresh values.
        }
        workgroupBarrier();
    }
}
