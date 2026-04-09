# Repository Reorganization Proposal

## 1) Fairness note on your benchmark question

- Your friend did not get the better score by lowering `N` in Part A (both branches used `N = 4096`).
- The comparison still had one fairness caveat: timing policy differed (`repeats=20` vs `repeats=3`, both using best/min timing).
- For publication-grade fairness, both kernels should be run under the same harness and same timing policy.

## 2) Proposed clean structure

Use one clear root for source code and one for outputs/docs.

```text
codesign/
  README.md
  requirements.txt
  .gitignore

  src/
    kernels/
      matmul_kernels.cl
    part_a/
      benchmark_part_a.py
      autotune_part_a.py
    part_b/
      benchmark_multidev.py
    common/
      opencl_utils.py
      benchmark_utils.py

  results/
    part_a/
      benchmark_latest.md
      autotune_results.csv
      autotune_results.json
    part_b/
      multidev_latest.md

  reports/
    lab1_report.md
    device_inventory.md

  experiments/
    yours/
      kernels_experimental.cl
      notes.md
    colleague/
      branch_partieA_snapshot/
        kernels_colleague.cl
        benchmark_colleague.py
      notes.md

  legacy/
    part1.py
    part11.py
    part12.py
    matmult.py
    final.py
    final1.py
    final_B.py
    test_opencl.py
```

## 3) Suggested rename map (current -> proposed)

- `kernels/kernels.cl` -> `src/kernels/matmul_kernels.cl`
- `part_A/benchmark.py` -> `src/part_a/benchmark_part_a.py`
- `part_A/k9_autotune.py` -> `src/part_a/autotune_part_a.py`
- `part_B/multidev.py` -> `src/part_b/benchmark_multidev.py`
- `LAB1_OPENCL_REPORT_AND_METHOD.md` -> `reports/lab1_report.md`
- `opencl_devices_result.md` -> `reports/device_inventory.md`
- `part_A/k9_autotune_results.csv` -> `results/part_a/autotune_results.csv`
- `part_A/k9_autotune_results.json` -> `results/part_a/autotune_results.json`

## 4) How to separate your work from your colleague

Use code ownership by folder, not by branch only:

- Keep your production candidate in `src/`.
- Keep colleague snapshot under `experiments/colleague/` (read-only reference).
- Keep your trial kernels under `experiments/yours/`.
- Only copy proven kernels from experiments into `src/` after correctness checks.

This removes ambiguity and avoids accidental overwrites.

## 5) No-conflict merge workflow (recommended)

1. Ensure your current branch is committed or stashed.
2. Create integration branch from your branch:
    - `git checkout -b integrate-colleague-partA`
3. Bring colleague files into a separate folder instead of direct overwrite:
    - `git checkout origin/partieA -- kernels/kernels.cl part_A/benchmark.py`
    - Move them to:
        - `experiments/colleague/branch_partieA_snapshot/kernels_colleague.cl`
        - `experiments/colleague/branch_partieA_snapshot/benchmark_colleague.py`
4. Add a single shared benchmark harness in `src/part_a/benchmark_part_a.py` that can compile both kernels and compare with identical settings.
5. Promote only the winner into `src/kernels/matmul_kernels.cl`.
6. Commit in small steps:
    - Commit A: structure + file moves (no code logic changes)
    - Commit B: colleague snapshot import
    - Commit C: unified benchmark harness + final chosen kernel

## 6) Immediate practical next step

Do a structure-only refactor first (moves/renames only), then run tests. This keeps diffs reviewable and makes grading safer.
