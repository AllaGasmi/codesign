# TP OpenCL Results Summary - Filled (Current Run)

Source: TP_OpenCL_Results_summary.pdf

## Header

- Program: CODESIGN - GL3
- Lab: LAB 1 - OpenCL Programming
- Deadline (results summary): April 9

## Team

- Students (max 4):
    -
    -
    -
    -
- Group:

## Hardware

- NVIDIA GPU: RTX 3050 Laptop GPU
- Integrated GPU: Intel Iris Xe Graphics

## Part A - Matrix Multiplication Kernel Optimization

| Method                       | Performance (GFLOPS/s) | Speedup vs Coalesced |
| ---------------------------- | ---------------------: | -------------------: |
| Coalesced                    |                 444.85 |                 1.00 |
| Technique giving best result |                4047.46 |                 9.10 |

## Part B - Multi-Device OpenCL (N = 8192)

| Configuration                                          | Performance (GFLOPS/s) | Speedup |
| ------------------------------------------------------ | ---------------------: | ------: |
| Only NVIDIA (Uncoalesced kernel)                       |                 111.18 |    1.00 |
| NVIDIA (Uncoalesced) + Integrated GPU (best technique) |                 327.00 |    2.94 |

## Submission reminder

Use this sheet for concise numerical reporting.
The detailed report is submitted separately.

## Measurement policy used

- Part A policy: N=4096, warmup=3, repeats=10, metric=min(event time), correctness check at N=1024.
- Part B policy: N=8192, work-group 16x16, split search across candidate row partitions.
