# LAB 1 - OpenCL Programming (Submission Draft)

## Team

- Students (max 4): [Fill names]
- Group: [Fill group]
- Date: 2026-04-09

## Hardware and OpenCL Devices

### NVIDIA CUDA Platform

- Device: NVIDIA GeForce RTX 3050 Laptop GPU
- Global memory: 3.9996 GB
- Global cache: 448 KB (READ_WRITE_CACHE)
- Cache line: 128 B
- Local memory: 48 KB (LOCAL)
- Constant memory: 64 KB
- Compute units: 16
- Max work-group size: 1024
- Max work-item size: [1024, 1024, 64]
- Lockstep unit (warp/wave): 32

### Intel OpenCL HD Graphics Platform

- Device: Intel(R) Iris(R) Xe Graphics
- Global memory: 9.4832 GB
- Global cache: 768 KB (READ_WRITE_CACHE)
- Cache line: 64 B
- Local memory: 64 KB (LOCAL)
- Constant memory: 4194296 KB
- Compute units: 80
- Max work-group size: 256
- Max work-item size: [256, 256, 256]
- Lockstep unit: 64

### Intel OpenCL Platform (CPU)

- Device: 12th Gen Intel(R) Core(TM) i5-12500H
- Global memory: 23.7079 GB
- Global cache: 1280 KB (READ_WRITE_CACHE)
- Cache line: 64 B
- Local memory: 256 KB (GLOBAL)
- Constant memory: 128 KB
- Compute units: 16
- Max work-group size: 8192
- Max work-item size: [8192, 8192, 8192]
- Lockstep unit: 128

Source of device info: device_inventory.md

---

## Part A - Matrix Multiplication Kernel Optimization

### Goal

Starting from COALESCED as reference, propose improvements and measure performance gain:

Gain = Performance(proposed kernel) / Performance(COALESCED)

### Benchmark setup

- Matrix size: N = 4096
- Data type: float32
- Reference work-group: 16x16
- Measured metric: GFLOPS from OpenCL profiling events

### Results (NVIDIA RTX 3050)

| Kernel |                      Technique | Time (ms) |  GFLOPS | Gain vs K2 |
| ------ | -----------------------------: | --------: | ------: | ---------: |
| K1     |                 NAIVE baseline |   1187.51 |  115.74 |      0.26x |
| K2     |          COALESCED (reference) |    312.67 |  439.56 |      1.00x |
| K3     |          TILING (local memory) |    237.51 |  578.67 |      1.32x |
| K4     |        Work per thread (WPT=4) |    167.08 |  822.57 |      1.87x |
| K5     | Transposed + rectangular tiles |     60.93 | 2255.61 |      5.13x |
| K6     |           2D register blocking |     60.40 | 2275.42 |      5.18x |
| K7     |        float4 vectorized loads |     54.90 | 2503.55 |      5.70x |
| K8     |     Wider loads + 2D registers |     54.95 | 2501.12 |      5.69x |
| K9     | Best (prefetch + 2D registers) |     54.00 | 2545.33 |      5.79x |
| K10    |       Tuned K9 + pragma unroll |     53.94 | 2547.89 |      5.80x |

Additional gain:

- K9 vs K1: 21.95x
- K10 vs K9: 1.001x (about +0.10%)
- K10 numerical correctness vs K2: max_abs_diff = 0.000000

### Minimum 2 proposed improvements (with explanation)

1. Local memory tiling (K3)

- Idea: load sub-tiles of A and B into \_\_local memory once, then reuse many times.
- Why it helps: lower global memory traffic, higher arithmetic intensity.
- Measured gain: 1.31x vs COALESCED.

2. Increased work per thread / register reuse (K4, K6)

- Idea: each thread computes multiple outputs and keeps partial sums in registers.
- Why it helps: fewer redundant loads and better instruction-level reuse.
- Measured gain: up to 5.17x (K6) vs COALESCED.

3. Vectorized memory operations (K7)

- Idea: use float4 loads/stores to increase useful data per memory instruction.
- Why it helps: better memory throughput and reduced instruction overhead.
- Measured gain: 5.70x vs COALESCED.

4. Prefetch + 2D blocking (K9)

- Idea: overlap tile loading with computation and keep 2D register blocking.
- Why it helps: improved pipeline utilization and better data reuse.
- Measured gain: 5.79x vs COALESCED (best result).

### Systematic Parameter Autotuning (K_TUNE)

We added a dedicated autotuner script using a parametric K9 structure with compile-time defines via build options:

-DTSM=... -DTSN=... -DTSK=... -DWPTM=... -DWPTN=... (plus derived RTSM/RTSN)

For each valid configuration we used:

- 3 warmup launches
- 5 timed launches
- mean GFLOPS from OpenCL event profiling

Constraint filters applied before running:

- local memory <= 45000 bytes
- threads per work-group in [64, 512]
- integer LPTA and LPTB
- divisibility constraints for N

Correctness policy:

- Every measured configuration was compared against K2 output.
- Only exact-correct results (max_abs_diff = 0.0) were eligible as winners.

#### Step B: all valid configurations (sorted by GFLOPS)

| TSM | TSN | TSK | WPTM | WPTN | WG (RTSMxRTSN) | Mean GFLOPS |   max_abs_diff | Exact |
| --: | --: | --: | ---: | ---: | -------------: | ----------: | -------------: | ----: |
|  64 |  64 |  32 |    8 |    8 |            8x8 |     3003.35 |        742.932 |    no |
|  64 |  64 |  16 |    8 |    8 |            8x8 |     2956.46 |        724.119 |    no |
| 128 | 128 |  16 |    8 |    8 |          16x16 |     2572.10 |          0.000 |   yes |
|  64 |  64 |   8 |    8 |    8 |            8x8 |     2043.77 |          0.000 |   yes |
|  64 |  64 |  32 |    4 |    4 |          16x16 |     1977.86 |        470.171 |    no |
| 128 | 128 |   8 |    8 |    8 |          16x16 |     1936.21 | 2962880512.000 |    no |
|  64 | 128 |  16 |    4 |    8 |          16x16 |     1618.56 |          0.000 |   yes |
| 128 |  64 |  16 |    8 |    4 |          16x16 |     1588.01 |          0.000 |   yes |
|  64 |  64 |  16 |    4 |    4 |          16x16 |     1510.40 |          0.000 |   yes |
|  64 |  64 |   8 |    4 |    4 |          16x16 |     1089.27 | 2962516224.000 |    no |

Note:

- The two fastest raw Step B configs were incorrect (non-zero max_abs_diff), so they were rejected.

#### Step C: independent tweaks on top-3 exact-correct Step B configs

| Base config     | Tweak                | Mean GFLOPS | max_abs_diff | Exact |
| --------------- | -------------------- | ----------: | -----------: | ----: |
| 128/128/16, 8/8 | fast_math            |     2566.07 |        0.000 |   yes |
| 128/128/16, 8/8 | Bsub padding (TSK+2) |     2243.87 |        0.000 |   yes |
| 128/128/16, 8/8 | pragma_unroll        |     2586.60 |        0.000 |   yes |
| 64/64/8, 8/8    | fast_math            |     2056.99 |        0.000 |   yes |
| 64/64/8, 8/8    | Bsub padding (TSK+2) |     1934.14 |        0.000 |   yes |
| 64/64/8, 8/8    | pragma_unroll        |     2053.23 |        0.000 |   yes |
| 64/128/16, 4/8  | fast_math            |     1632.69 |        0.000 |   yes |
| 64/128/16, 4/8  | Bsub padding (TSK+2) |     1602.55 |        0.000 |   yes |
| 64/128/16, 4/8  | pragma_unroll        |     1627.19 |        0.000 |   yes |

#### Step D: volatile test on best Step B+C config

| Config          | Variant     | Mean GFLOPS | max_abs_diff | Exact |
| --------------- | ----------- | ----------: | -----------: | ----: |
| 128/128/16, 8/8 | volatile_id |     2375.75 |        0.000 |   yes |

Best exact-correct from B+C+D:

- TSM=128, TSN=128, TSK=16, WPTM=8, WPTN=8
- Tweak: pragma_unroll
- Mean GFLOPS: 2586.60
- Gain vs K2 (autotune run): 5.88x

### K10 - Combined Ultra Kernel (adopted)

From the sweep, the winning exact-correct setup was integrated as K10:

- K9 structure retained (no algorithm change)
- Hardcoded winning parameters
- Explicit unroll pragmas on accumulation loops

Reason this wins:

- Preserves the proven K9 memory and dataflow behavior
- Reduces loop overhead/instruction scheduling pressure
- Avoids incorrect high-throughput parameter points

Final benchmark harness result (part_A/benchmark.py):

- K9: 2545.33 GFLOPS
- K10: 2547.89 GFLOPS
- K10 vs K9: 1.001x
- K10 correctness: max_abs_diff = 0.0

---

## Part B - Running Kernel on Multiple OpenCL Devices

### Required setup from assignment

- N = 8192
- NVIDIA device executes UNCOALESCED/NAIVE kernel
- Integrated GPU executes best method from Part A
- Evaluate speedup:

Speedup = Performance(2 devices) / Performance(NVIDIA only)

### Step 1 - Individual full-matrix performance (N=8192)

- NVIDIA RTX 3050 with NAIVE: 110.84 GFLOPS
- Intel Iris Xe with BEST (K9): 434.10 GFLOPS

### Step 2 - Split strategy and justification

We split output matrix C by rows:

- NVIDIA computes top M0 rows
- Intel computes remaining N - M0 rows

Theoretical first estimate:

- M0 approx N \* GF_nvidia / (GF_nvidia + GF_intel)
- M0 approx 8192 \* 110.84 / (110.84 + 434.10) approx 1664 rows

Then we auto-tune around this estimate and across coarse candidates.

### Tested row splits and measured performance

| NVIDIA rows | Intel rows | Parallel GFLOPS | Speedup vs NVIDIA |
| ----------: | ---------: | --------------: | ----------------: |
|         512 |       7680 |          346.43 |             3.13x |
|         768 |       7424 |          324.79 |             2.93x |
|        1024 |       7168 |          286.97 |             2.59x |
|        1280 |       6912 |          277.34 |             2.50x |
|        1408 |       6784 |          278.93 |             2.52x |
|        1536 |       6656 |          270.01 |             2.44x |
|        1664 |       6528 |          260.49 |             2.35x |
|        1792 |       6400 |          256.70 |             2.32x |
|        1920 |       6272 |          249.79 |             2.25x |
|        2048 |       6144 |          240.31 |             2.17x |
|        2304 |       5888 |          238.39 |             2.15x |
|        2560 |       5632 |          228.73 |             2.06x |
|        2816 |       5376 |          218.87 |             1.97x |
|        3072 |       5120 |          209.12 |             1.89x |

### Final best result (Part B)

- Best split: NVIDIA 512 rows, Intel 7680 rows
- Parallel wall time: 3362.47 ms
- Parallel throughput: 327.00 GFLOPS
- Speedup vs NVIDIA-only baseline: 2.94x

Interpretation:

- Because NVIDIA is forced to run NAIVE, it becomes the bottleneck if assigned too many rows.
- Best total throughput is obtained by assigning most rows to Intel BEST kernel.

---

## What We Did to Achieve These Results (Method Summary)

1. Baseline measurement and profiling

- Measured NAIVE and COALESCED first to establish references.
- Kept profiling events enabled and used minimum of repeated runs.

2. Iterative kernel optimization for Part A

- Added local-memory tiling.
- Added register blocking and work-per-thread.
- Added transposed/rectangular tiling.
- Added vectorized float4 loads.
- Added prefetch strategy for final best kernel (K9).

3. Multi-device execution design for Part B

- Split matrix by row ranges so each device computes disjoint output rows.
- Kept required assignment policy: NVIDIA=NAIVE, Intel=BEST.

4. Robust engineering fixes to make Part B reproducible

- Disabled PyOpenCL cache in script for Windows stability.
- Avoided unstable cross-context waits by waiting each event directly.
- Used chunked row-split execution and tested many split candidates.

5. Auto-tuning for best speedup

- Computed theoretical split from single-device GFLOPS.
- Ran a candidate sweep and selected split with highest measured GFLOPS.

6. Systematic parameter autotuning for Part A

- Built a parametric K9 (K_TUNE) using compile-time define sweep over TSM/TSN/TSK/WPTM/WPTN.
- Ran all valid configurations under constraints with 3 warmup + 5 timed iterations.
- Enforced exact correctness (max_abs_diff = 0.0) before selecting winners.

7. K10 exploration and selection policy

- Promoted the best exact-correct K_TUNE setup into K10.
- Kept K10 only after verifying measurable gain and max_abs_diff = 0.0.

---

## Team Parallel Comparison (Fair Policy, Same Timing Rules)

To compare both team workstreams fairly, we used one shared benchmark harness with identical policy:

- N = 4096
- warmup = 3
- repeats = 10
- metric = min OpenCL event time
- correctness check at N = 1024 versus coalesced output

Measured values:

| Workstream | Coalesced (GFLOPS) | Best (GFLOPS) | Best/Coalesced | max_abs_diff |
| ---------: | -----------------: | ------------: | -------------: | -----------: |
|   Main src |             444.85 |      2585.19  |          5.81x |          0.0 |
|   Teammate snapshot |    443.61 |      4047.46  |          9.12x |          0.0 |

Conclusion from fair policy:

- Both implementations are numerically correct under the check used.
- The teammate snapshot currently provides a significantly higher Part A throughput under identical benchmark policy.

---

## Files Used

- Part A benchmark: src/part_a/benchmark_part_a.py
- Fair team comparison: src/part_a/fair_compare_part_a.py
- Part B benchmark: src/part_b/benchmark_multidev.py
- Shared kernels: src/kernels/matmul_kernels.cl
- Teammate snapshot kernels: experiments/colleague/branch_partieA_snapshot/matmul_kernels_colleague.cl
- Device characterization: reports/device_inventory.md

---

## Ready-to-fill values for TP summary sheet

A - Matrix Multiplication Kernel Optimization

- COALESCED performance: 444.85 GFLOPS
- Technique giving best result: team best kernel from parallel track comparison
- Best performance: 4047.46 GFLOPS
- Speedup (best / coalesced): 9.10x

B - Running on multiple OpenCL devices (N=8192)

- Only NVIDIA UNCOALESCED: 111.18 GFLOPS
- NVIDIA UNCOALESCED + Integrated BEST: 327.00 GFLOPS
- Speedup: 2.94x
