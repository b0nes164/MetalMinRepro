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
2025-04-30 09:29:56.283 metalMinRepro[79282:11031643] Test failed: got 1024 at 33186
2025-04-30 09:29:56.283 metalMinRepro[79282:11031643] Batch 146: Scan buffer validation FAILED.
```

## Repro details

We ran this test on two different MacBooks:

- MacBook Pro, 16-inch 2021, M1 Max. Sequoia 15.4.1.
- MacBook Air, 13-inch 2024, M3. Sequoia 15.4.1.

M3 tests completed quickly (~3 ms per test), and always correctly.
M1 tests run much slower (~600--3000 ms per test) and intermittently fail, and when they do, they fail in bunches.

## What we think

So to recap, the starvation on the M1 Max is not stressing the hardware to the point that a failure emerges, but instead, the starvation is causing the watchdog to kill the shader---albeit while allowing the program to coninue running.
