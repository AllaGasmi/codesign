# OpenCL Matrix Multiplication for Codesign Lab 1

Suggested repository name:

- codesign-opencl-lab1-matmul

Suggested repository topics/tags:

- opencl
- gpu-computing
- matrix-multiplication
- pyopencl
- performance-optimization
- heterogeneous-computing
- nvidia
- intel-gpu

## Project Purpose

This repository contains Lab 1 work for OpenCL programming:

- Part A: optimize matrix multiplication kernels and measure speedups.
- Part B: run matrix multiplication on two OpenCL devices and maximize speedup.

The repository is organized to keep production code, experiments, and reports separate while allowing your implementation and your colleague snapshot to coexist on the same branch.

## Repository Structure

- src: production candidate code
    - src/kernels/matmul_kernels.cl
    - src/part_a/benchmark_part_a.py
    - src/part_a/autotune_part_a.py
    - src/part_a/fair_compare_part_a.py
    - src/part_b/benchmark_multidev.py
- experiments: isolated comparisons and side work
    - experiments/colleague/branch_partieA_snapshot/
    - experiments/yours/
- results: benchmark/autotuning outputs
    - results/part_a/
- reports: report artifacts and assignment context
    - reports/lab1_report.md
    - reports/device_inventory.md
    - reports/context/
- legacy: older scripts kept for traceability

## Assignment Context in English

- Assignment translation: reports/context/lab1_assignment_en.md
- Results sheet translation: reports/context/results_summary_template_en.md
- Professor classroom addendum (translated): reports/context/professor_additional_notes_en.md

## Quick Start

Prerequisites:

- Python 3.10+
- pyopencl
- numpy
- OpenCL drivers for your GPUs

Typical runs:

- Part A benchmark: python src/part_a/benchmark_part_a.py
- Part A fair compare (team parallel tracks under identical timing policy):
  python src/part_a/fair_compare_part_a.py --n 4096 --check-n 1024 --warmup 3 --repeats 10
- Part B benchmark: python src/part_b/benchmark_multidev.py

## Coexistence Model (Team Parallel Work)

- Your active implementation stays under src.
- Colleague implementation is frozen under experiments/colleague/branch_partieA_snapshot.
- Promotion rule: only copy from experiments into src after correctness and fair benchmark validation.

This avoids accidental overwrite and keeps review clean.

## Current Status Summary

See reports/context/work_status_and_ownership.md for:

- what has been done in this branch,
- what came from your implementation work,
- what is imported from colleague snapshot,
- what remains before final submission.
