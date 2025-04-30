# Repro for M1 Max issues with chained communication between workgroups

## Background

We are implementing compute primitives on the GPU. Specifically, we are using a "chained" formulation of the parallel primitive "scan" (prefix sum). The primary data structure here is an array in global memory with size equal to the number of workgroups. Each workgroup "owns" one entry in this array. Each entry in the array is marked (in bits 31 and 30) as NOT_READY (the initial value), READY, or INCLUSIVE.

- If entry n is READY, then entry n contains the partial result computed by workgroup n.
- If entry n is INCLUSIVE, then entry n contains the partial result that combines the results of workgroups 0--n, inclusive.

When every entry is INCLUSIVE, we have completed the computation.

In the chained formulation, individual workgroup n:

- Independently computes a part of the solution
- Posts its partial solution to global memory, marking it as READY. (If n == 0, we mark it as INCLUSIVE instead.)
- Sets a "lookback_id" to entry n-1.
- Loop:
  - Fetch entry lookback_id.
  - If that entry is INCLUSIVE, combine it locally with my partial solution, and post that to entry n tagged with INCLUSIVE. Return.
  - If that entry is READY, combine it locally with my partial solution, decrement lookback_id, and jump to the top of the loop

In the test we are submitting, the "partial solution" is the integer 1024, and "combine" is add. So we expect after our implementation completes, array entry n contains the value 1024 \* n.

### More grungy technical details

We lied about the structure of the array above. It's actually an array of TWO u32s per workgroup. We split the 32b value that we wish to store across the two u32s (in bits 15:0) and store the flag values in bits 31:30. We choose this structure because we wish to use all 32 bits of the value in our computations and can't store both a 32b value and 2 bits of flags in one u32. References to `split` and `join` in the source code are implementing this structure. We have not seen any issues with our use of this structure.

## Building and running the test

The below run is on a M3.

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

The test could fail in multiple ways and will print an error that looks like what's below. We have seen one specific failure mode on M1, "Scan buffer validation".

```
2025-04-30 09:29:56.283 metalMinRepro[79282:11031643] Test failed: got 1024 at 33186 (flags: 0x40000000, 0x40000000)
2025-04-30 09:29:56.283 metalMinRepro[79282:11031643] Batch 146: Scan buffer validation FAILED.
```

## Repro details

We ran this test on two different MacBooks:

- MacBook Pro, 16-inch 2021, M1 Max. Sequoia 15.4.1.
- MacBook Air, 13-inch 2024, M3. Sequoia 15.4.1.

M3 tests completed quickly (~3 ms per test), and always correctly.
M1 tests run much slower (~600--12000 ms per test) and intermittently fail, and when they do, they fail in bunches.

## What this specific failure indicates

`Test failed: got 1024 at 33186 (flags: 0x40000000, 0x40000000)`

Workgroup 33186 posted its part of the solution (1024) and marked it as READY (0x40000000). However, neither this workgroup (nor any later workgroup) ever enter the lookback loop and update this value.

## What we think

We believe that the failure indicates that workgroups are resident on the GPU but are not run; they are being blocked by other workgroups. In a word, they are starved. The starvation does not stress the hardware to the point where we see a failure, but instead the starvation eventually causes a watchdog to kill the shader (but allows the CPU program to continue running).
