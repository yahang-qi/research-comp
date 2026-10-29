# Reproducing Figures 2 & 3 of Zhao et al. (2012, JASA) — OWL

Examples 1-4 are the paper's; **Example 5** is an added scenario with
delta(X) = 3(X1^2 + X2^2 - 2/3) (optimal rule: treat A = 1 outside the circle X1^2 + X2^2 = 2/3),
run with P = 3 covariates (X1, X2, X3 ~ U[-1,1]); the response model
Y = 1 + 2X1 + X2 + 0.5X3 + delta(X)A + N(0,1) is the same for all examples.

Files
- `R/owl_sim_functions.R` – data generation, the methods (OWL-Gaussian / OWL-Linear / OWL-Poly via `DynTxRegime::owl`, l1-PLS via `glmnet`, OLS), evaluation metrics, and a run-time patch for a DynTxRegime 4.16 bug (non-linear-kernel OWL drops the sign y_i from the kernel coefficients). `OWL_P` (env var, default 50) sets the number of covariates.
- `scripts/run/run_simulation.R` – runs all replications in parallel (PSOCK cluster, load-balanced over scenario x n x replicate) and checkpoints blocks of 20 reps to `OUT_DIR` (default `outputs/results/ex1-4/`). Env vars: `SCENARIOS` (default `1,2,3,4`), `OUT_DIR`, `OWL_P`.
- `scripts/run/run_poly.R` – OWL with a degree-2 polynomial kernel (1 + x'x')^2 (same seeds, validation sets and lambda grid) for `SCENARIOS` (default `5`), saved as `poly2.rds` in `OUT_DIR` (default `outputs/results/g5_poly/`).
- `scripts/sbatch/run_simulation.sh` – Slurm job array for examples 1-4; `scripts/sbatch/run_g5.sh` / `scripts/sbatch/run_g5_poly.sh` – Slurm jobs for Example 5 (`OWL_P=3`, 1000 reps; `run_g5.sh` is a 50-task array, one 20-rep block each).
- `scripts/plot/` – all plotting code; figures go to `outputs/figures/` (PDF only). Run from the project root:
  - `scripts/plot/plot_figures.R` – reads the raw `.rds` results from `OUT_DIR` (default `outputs/results/ex1-4/`; a comma-separated list of folders is allowed), writes `Figure2_MSE_value<FIG_TAG>.pdf` and `Figure3_MSE_misclassification<FIG_TAG>.pdf`, and prints the summary table (no CSV files). OWL-Poly2 is drawn when its results are included. Example 5: `OUT_DIR=outputs/results/g5,outputs/results/g5_poly FIG_TAG=_g5 Rscript scripts/plot/plot_figures.R`.
  - `scripts/plot/plot_figures_g5.R` – Example 5 only: MSE of policy value and MSE of misclassification as two panels with one shared legend → `Figure2_3_g5.pdf` (reads `outputs/results/g5` and `outputs/results/g5_poly`; `Rscript scripts/plot/plot_figures_g5.R`).
  - `scripts/plot/plot_decision_boundary.R` – true decision boundary D* of Examples 1-4 → `Decision_boundaries.pdf`; `scripts/plot/plot_decision_boundary_g5.R` – Example 5 → `Decision_boundary_g5.pdf`.
  - `scripts/plot/plot_estimated_boundary_g5.R [n]` – Example 5: estimated vs optimal boundary for d = 3, 10, 20, 50 covariates (one n = 800 training set drawn with 50 covariates, nested subsets, main-loop tuning grids) → `Estimated_boundaries_g5.pdf`. Run `OWL_P=50 Rscript scripts/plot/plot_estimated_boundary_g5.R` or `sbatch scripts/sbatch/run_estimated_boundary_g5.sh`.
  - `scripts/plot/plot_estimated_boundary.R [n]` – estimated boundaries of l1-PLS and OWL-Gaussian for Examples 1-4 (assumes P = 50; ~4 min).

Repository layout (everything generated lives in `outputs/`, which is not tracked by Git):

```
R/                    owl_sim_functions.R  (library code, sourced by the scripts)
scripts/run/          simulation drivers
scripts/plot/         plotting scripts
scripts/sbatch/       Slurm job scripts
outputs/results/      ex1-4/  g5/  g5_poly/   (raw .rds results)
outputs/figures/      PDF figures
outputs/logs/         Slurm output (the folder must exist before `sbatch`; kept by .gitkeep)
```

All `scripts/` must be launched from the project root (they `source("R/owl_sim_functions.R")` and write to `outputs/` relative to it), e.g. `sbatch scripts/sbatch/run_g5.sh`.

Run (Terminal, in this folder). First argument = replications (default 200),
second = parallel workers (default: all cores but one); e.g. `1000 8` for the paper's setting.
```
Rscript scripts/run/run_simulation.R 200
Rscript scripts/plot/plot_figures.R
```
Missing packages (DynTxRegime, glmnet) are installed automatically from CRAN.
Interrupt any time; re-running resumes from the last finished block, and
`scripts/plot/plot_figures.R` can be run on partial results.

Cost: n = 800 dominates (about 45 s per replicate per core for the CV grid of OWL with P = 50).
