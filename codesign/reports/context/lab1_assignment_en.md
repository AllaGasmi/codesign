# Codesign Lab 1 Assignment (OpenCL) - English Reference

Source: Codesign_Lab1_OpenCL_2026 (1).pdf

## Scope

This lab focuses on optimizing OpenCL matrix multiplication and then extending execution across multiple OpenCL devices.

The final report should be concise and rely mostly on figures and visual explanations, with minimal text.

## Submission-relevant constraints

- Part A uses coalesced kernel performance as the reference baseline.
- Part B targets N = 8192 with work-group size 16 x 16 and a 2-GPU split.
- Part B speedup is defined against NVIDIA-only performance.

## Device Inventory Task

Provide characteristics of OpenCL-compatible devices available on your PC:

- Device reference/model
- Global, local, and cache memory sizes
- Number of compute units (Streaming Multiprocessors)
- Maximum work-items and work-groups
- Other relevant execution limits

## Part A - Matrix Multiplication Kernel Optimization

Starting from the performance of the coalesced kernel as reference:

- Propose at least 2 optimization improvements.
- For each optimization:
    - Explain the technique.
    - Explain why and how it improves performance.
    - Implement the OpenCL kernel.
    - Report performance gain as:

Gain = Proposed Technique Performance / Coalesced Performance

Recommended reading:

- https://cnugteren.github.io/tutorial/pages/page1.html
- Tutorial: OpenCL SGEMM tuning

## Part B - Running the Kernel on Multiple OpenCL Devices

For one OpenCL device, total processing time is:

- Host-to-device transfer
- Kernel execution
- Device-to-host transfer

On systems with multiple OpenCL devices, matrix multiplication can be split and executed in parallel.

This lab specifically uses two GPUs:

- Integrated GPU
- NVIDIA dedicated GPU

### Objective

Speed up the NVIDIA uncoalesced implementation for:

- N = 8192
- Work-group size = 16 x 16

Execution split:

- NVIDIA GPU computes one part of C using uncoalesced access.
- Integrated GPU computes the remaining part of C using the best-performing method from Part A.

Speedup metric:

Speedup = Performance in GFLOPS/s (2 devices) / Performance in GFLOPS/s (NVIDIA only)

### Required Deliverables in Part B

1. Give standalone whole-matrix (8192) performance for both GPUs:

- NVIDIA with uncoalesced method
- Integrated GPU with best method from Part A

2. Explain and justify the split strategy:

- Why the chosen sub-matrix sizes per device are appropriate

3. Provide host-side OpenCL code and final speedup result

## Notes on terminology in original PDF

The original assignment text contains spelling variants such as:

- coalsced / coalesced
- uncoalsced / uncoalesced

This markdown keeps technical meaning unchanged while standardizing wording in English.
