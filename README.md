# Repro for Stalled workgroup(s) in chained communication test, M1 but not M3/M4

tl;dr: On M1, we demonstrate a compute shader that intermittently exhibits a stalled workgroup that leads to a timeout. This behavior does not occur on M3 or M4.

## Background

We are implementing compute primitives on the GPU. Specifically, we are using a "chained" formulation of the parallel primitive "scan" (prefix sum). (FWIW, this formulation is the foundation of the fastest scan (prefix-sum) and sort implementations on GPUs. The relevant academic publication is [Merrill and Garland, 2016](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back).) The primary data structure here is an array in global memory (the "scan buffer") with size equal to the number of workgroups. Each workgroup "owns" one entry in this array. Each entry in the array is marked (in bits 31 and 30) as NOT_READY (the initial value), READY, or INCLUSIVE.

- If entry n is READY, then entry n contains the partial result computed by workgroup n.
- If entry n is INCLUSIVE, then entry n contains the partial result that combines the results of workgroups 0--n, inclusive.

When every entry in the scan buffer is INCLUSIVE, we have completed the computation.

In the chained formulation, individual workgroup n:

- Independently computes a part of the solution
- Posts its partial solution to its corresponding entry in the scan buffer, marking it as READY. (If n == 0, we mark it as INCLUSIVE instead.)
- Sets a "lookback_id" to entry n-1.
- Loop:
  - Fetch entry lookback_id from the scan buffer.
  - If that entry is INCLUSIVE, combine it locally with my partial solution, and post that to entry n in the scan buffer tagged with INCLUSIVE. Return.
  - If that entry is READY, combine it locally with my partial solution, decrement lookback_id, and jump to the top of the loop

In the test we are submitting, the "partial solution" is the integer 1024, and "combine" is add. So we expect after our implementation completes, entry n in the scan buffer contains the value 1024 \* n.

Note that this chained structure imposes a serial dependency across all workgroups: workgroup n depends on workgroup n-1. However, in practice, the scan buffer resides in cache and the resolution of the dependency (one atomic read, one addition, one atomic store) is fast; this serial dependency is not the bottleneck of a chained-scan or chained-sort ("Onesweep") implementation.

### More grungy technical details

We lied about the structure of the scan buffer above. It's actually an array of TWO u32s per workgroup. We split the 32b value that we wish to store across the two u32s (in bits 15:0) and store the flag values in bits 31:30. We choose this structure because we wish to use all 32 bits of the value in our computations and can't store both a 32b value and 2 bits of flags in one u32. References to `split` and `join` in the source code are implementing this structure. We have not seen any issues on Apple hardware with our use of this structure. However, it was tricky to get right from our perspective and might be an interesting future internal test for you to use on your hardware.

## Building and running the test

The below run is on a M3 (and displays the expected behavior).

```
% make
xcrun metal initShader.metal -o initShader.metallib
xcrun metal stressShader.metal -o stressShader.metallib
clang++ -fmodules -framework CoreGraphics main.m -o metalMinRepro
% time ./metalMinRepro 10000
10000 / 10000 ALL TESTS PASSED
2025-04-30 10:07:22.496 metalMinRepro[39334:12670060] All batches completed.
./metalMinRepro 10000  2.68s user 1.18s system 11% cpu 33.421 total
```

The executable `metalMinRepro` takes one argument, the number of trials. It may take tens of minutes to reproduce the issue on M1. Start with 100 trials, then go to 1000 and do a few runs at 1000.

The test could fail in multiple ways and will print an error for each of them. We have seen two different kinds of errors on M1 but no errors on M3 or M4.

### Timeout failure

After submitting each command buffer, we check to see if the compute shader successfully executed.

We have seen one error code, timeout, only on the M1. This indicates starvation induced by a lack of forward progress guarantees.

```
2025-04-30 11:44:47.552 metalMinRepro[82313:11136212] Command buffer execution failed with error: Error Domain=MTLCommandBufferErrorDomain Code=1 "Internal Error (0000000e:Internal Error)" UserInfo={NSLocalizedDescription=Internal Error (0000000e:Internal Error), NSUnderlyingError=0x600002740630 {Error Domain=IOGPUCommandQueueErrorDomain Code=14 "(null)"}}
```

### Scan buffer validation failure

We look at the results in our scan buffer array.

We have seen one specific failure mode on M1, "Scan buffer validation".

```
2025-04-30 09:29:56.283 metalMinRepro[79282:11031643] Test failed: got 1024 at 33186 (flags: 0x40000000, 0x40000000)
2025-04-30 09:29:56.283 metalMinRepro[79282:11031643] Batch 146: Scan buffer validation FAILED.
```

### Error buffer validation failure

The test program also checks a dedicated "error buffer" that the stressShader can write to if it detects specific inconsistencies during its execution. This buffer allows for more granular error reporting from within the shader's logic. An error logged here would manifest as a specific message printed to the console by the CheckError function on the CPU side after the GPU work is completed. These errors would indicate the following:

- ERROR_TYPE_MESSAGE: This error is logged if a workgroup n, during its lookback phase (reading entry n-1, n-2, etc.), encounters a flag_payload in the scan buffer that does not conform to any of the valid, expected states (i.e., NOT_READY, a correctly formed READY state for that entry, or a correctly formed INCLUSIVE state). This could imply:

  - The workgroup whose entry is being read (the lookback_id entry) may have stalled or malfunctioned before it could update its scan buffer entry to a well-defined state.
  - A more subtle race condition or memory coherency issue might be leading to a corrupt or unexpected flag_payload being observed. This is a point of attention given our "grungy technical detail" of splitting a logical entry across two u32 values; an inconsistency here could manifest as this error.

- ERROR_TYPE_SHUFFLE_READY or ERROR_TYPE_SHUFFLE_INC: These errors are triggered if, after a workgroup reads what it believes to be valid flags and data from a lookback_id entry and combines this with its local partial sum, the resulting new partial sum (prev_red) is mathematically incorrect based on the expected values for that stage of the scan. This could point to:

  - Faulty `simd_shuffle` operations within our `join` function; investigating this is a primary diagnostic purpose for these error types. Such faults can occur if threads within a SIMD group (subgroup) take divergent execution paths due to branching and then fail to correctly reconverge before the `simd_shuffle` is called. An improperly executed shuffle directly leads to a corrupted `prev_red` value.
  - Corruption in the data portion (VALUE_MASK) of the flag_payload read from the scan buffer, even if the flags themselves appeared valid.

- ERROR_TYPE_SGSIZE: This error indicates a mismatch between the BLOCK_DIM (the subgroup size the shader was compiled expecting) and the actual subgroup size (sgSize) reported by the hardware at runtime. This is a fundamental configuration check performed at the beginning of the shader.

**We have not observed any of these errors on any device**.

## Repro details

We ran this test on three different MacBooks:

- MacBook Pro, 16-inch 2021, M1 Max. Sequoia 15.4.1.
- MacBook Air, 13-inch 2024, M3. Sequoia 15.4.1.
- Macbook Pro, MX2Y3LL/A, M4. Sequoia 15.4.1.

M4 and M3 tests completed quickly (~2.5 ms, ~3 ms per test), and always correctly.
M1 tests run much slower (~600--12000 ms per test) and intermittently fail, and when they do, they fail in bunches (many workgroups get an incorrect result). The large difference in runtime between M1 and M3/M4 is also indicative of the starvation problem.

## What this specific failure indicates

`Test failed: got 1024 at 33186 (flags: 0x40000000, 0x40000000)`

Workgroup 33186 posted its part of the solution (1024) and marked it as READY (0x40000000). However, neither this workgroup (nor any later workgroup) ever enter the lookback loop and update this value.

What we actually see on failure is a large number of individual incorrect results, e.g.:

```
2025-04-30 11:44:47.553 metalMinRepro[82313:11136212] Test failed: got 1024 at 23212 (flags: 0x40000000, 0x40000000)
2025-04-30 11:44:47.553 metalMinRepro[82313:11136212] Test failed: got 1024 at 23215 (flags: 0x40000000, 0x40000000)
2025-04-30 11:44:47.553 metalMinRepro[82313:11136212] Test failed: got 1024 at 23216 (flags: 0x40000000, 0x40000000)
2025-04-30 11:44:47.553 metalMinRepro[82313:11136212] Test failed: got 1024 at 23222 (flags: 0x40000000, 0x40000000)
2025-04-30 11:44:47.553 metalMinRepro[82313:11136212] Test failed: got 1024 at 23223 (flags: 0x40000000, 0x40000000)
...
```

and then eventually after many hundreds of "got 1024"s the value in the scan buffer stops being 1024 and starts being 0. (Below is a different run from the run above.)

```
2025-04-30 13:36:28.739 metalMinRepro[49493:223591] Test failed: got 1024 at 55755 (flags: 0x40000000, 0x40000000)
2025-04-30 13:36:28.739 metalMinRepro[49493:223591] Test failed: got 0 at 58476 (flags: 0x0, 0x0)
```

Entry n in the scan buffer can only be written by workgroup n. Thus the failures we see indicate the following:

- If we see 1024 in entry n ("got 1024"), it means workgroup n started and posted its READY value, but stalled (or was killed) before it could complete lookback
- If we see 0 in entry n ("got 0"), it means workgroup n did not start.

We believe we see these clustered failures because:

- A single workgroup is truly stalled. (This is the behavior we hope you can fix.)
- A watchdog timer sees this stall and kills the shader.
- All in-flight workgroups immediately stop, wherever they are.
- Some workgroups have completed their lookback (and have posted the correct result), but some have not; they have posted their local result (1024) but have not completed lookback to post the final (inclusive) result.

We know nothing about how a timeout affects a running kernel (e.g., does it discard all dirty values stored in cache?); the above results only reflect what we can see in the copy back of the scan buffer from GPU to CPU.

## What we think

We believe that the failure indicates that workgroups are resident on the GPU but are not run; they are being blocked by other workgroups. In a word, they are starved. The starvation does not stress the hardware to the point where we see a failure, but instead the starvation eventually causes a watchdog to kill the shader (but allows the CPU program to continue running).
