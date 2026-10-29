## =============================================================================
##  Simulation driver for Figures 2 & 3 of Zhao, Zeng, Rush & Kosorok (2012,
##  JASA), "Estimating Individualized Treatment Rules Using Outcome Weighted
##  Learning".
##
##  From a terminal, inside this folder:
##      Rscript scripts/run/run_simulation.R            # 200 replications, all cores - 1
##      Rscript scripts/run/run_simulation.R 1000 8     # paper's 1000 reps on 8 workers
##  Optional environment variables: SCENARIOS (default "1,2,3,4"), OUT_DIR (default
##  "outputs/results/ex1-4") and OWL_P (number of covariates, default 50), e.g.
##      OWL_P=3 SCENARIOS=5 OUT_DIR=outputs/results/g5 Rscript scripts/run/run_simulation.R 100 16
##  or in R / RStudio:  setwd("<this folder>"); source("scripts/run/run_simulation.R")
##
##  Replications are run in parallel on a PSOCK cluster (works on macOS,
##  Linux, Windows and inside RStudio).  Results are checkpointed in
##  OUT_DIR every BLOCK replications, so the run can be stopped and resumed;
##  plot_figures.R can be run at any time on whatever has finished.
## =============================================================================
library(parallel)

## ---- settings -----------------------------------------------------------------
args  <- suppressWarnings(as.integer(commandArgs(trailingOnly = TRUE)))
REPS  <- if (length(args) >= 1 && !is.na(args[1]) && args[1] >= 1) args[1] else 200L
SLURM_CPUS <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK")))
CORES <- if (length(args) >= 2 && !is.na(args[2]) && args[2] >= 1) args[2] else
         if (!is.na(SLURM_CPUS)) SLURM_CPUS else
           max(1L, parallel::detectCores() - 1L)
## Slurm job array: task k of N runs blocks k+1, k+1+N, ... (checkpoint files
## are per block, so array tasks never write the same file).
TASK_ID <- suppressWarnings(as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID")))
NTASKS  <- suppressWarnings(as.integer(Sys.getenv("SLURM_ARRAY_TASK_COUNT")))
if (is.na(TASK_ID) || is.na(NTASKS)) { TASK_ID <- 0L; NTASKS <- 1L }
BLOCK <- 20L                                # replicates per checkpoint
NS    <- c(30L, 100L, 200L, 400L, 800L)     # training sample sizes
NVAL  <- 10000L                             # validation sample size
SCEN  <- as.integer(strsplit(Sys.getenv("SCENARIOS", "1,2,3,4"), ",")[[1]])  # e.g. SCENARIOS=5
OUT   <- Sys.getenv("OUT_DIR", "outputs/results/ex1-4")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

## Tuning grids (5-fold CV maximising the estimated value, as in the paper).
##  c = 1 / lambda_n (paper's penalty parameter);  DynTxRegime lambda = c / n.
##  kparam = Gaussian bandwidth: k(x,x') = exp(-||x-x'||^2 / (2 kparam^2)).
##  With 50 U[-1,1] covariates E||x - x'||^2 = 33.3, so kparam in {2,4,8}
##  spans local-to-almost-linear kernels.
grid <- list(lambdas_rbf = c(0.5, 4, 32),
             kparam      = c(2, 4, 8),
             lambdas_lin = c(0.5, 4, 32))

## ---- packages -------------------------------------------------------------------
need <- c("DynTxRegime", "glmnet")
miss <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
if (length(miss)) install.packages(miss, repos = "https://cloud.r-project.org")

## ---- validation sets (one fixed set per scenario) -----------------------------------
source("R/owl_sim_functions.R")
vals <- lapply(SCEN, function(s) { set.seed(20120000 + s); gen_data(NVAL, s) })
names(vals) <- SCEN

## ---- cluster ----------------------------------------------------------------------
cat(sprintf("Running %d replications x %d scenarios x %d sample sizes on %d workers\n",
            REPS, length(SCEN), length(NS), CORES))
cl <- makeCluster(CORES)
clusterEvalQ(cl, {
  ## one BLAS thread per worker (avoid oversubscription with parallel workers)
  Sys.setenv(OPENBLAS_NUM_THREADS = "1", OMP_NUM_THREADS = "1",
             VECLIB_MAXIMUM_THREADS = "1")
  source("R/owl_sim_functions.R")
  NULL
})
clusterExport(cl, c("vals", "grid", "NS"))

run_task <- function(task) {
  s <- task$s; n <- task$n; r <- task$r
  set.seed(1e6 * s + 1e3 * match(n, NS) + r)       # reproducible per replicate
  out <- tryCatch(one_rep(n, s, vals[[as.character(s)]], grid),
                  error = function(e) NULL)
  if (!is.null(out)) { out$rep <- r; out$scenario <- s; out$n <- n }
  out
}

## ---- main loop: blocks of replicates, all (scenario, n) cells in parallel ------------
nblocks <- ceiling(REPS / BLOCK)
my_blocks <- seq_len(nblocks)[(seq_len(nblocks) - 1L) %% NTASKS == TASK_ID]
cat(sprintf("Array task %d of %d: blocks %s\n", TASK_ID, NTASKS,
            paste(my_blocks, collapse = ",")))
t_start <- Sys.time()
for (i in seq_along(my_blocks)) {
  b <- my_blocks[i]
  f <- file.path(OUT, sprintf("block%03d.rds", b))
  if (file.exists(f)) { cat("block", b, "already done - skipping\n"); next }
  reps  <- ((b - 1) * BLOCK + 1):min(b * BLOCK, REPS)
  tasks <- expand.grid(r = reps, s = SCEN, n = NS)
  tasks <- tasks[order(-tasks$n), ]                 # largest n first: load balance
  tasks <- lapply(seq_len(nrow(tasks)), function(i) as.list(tasks[i, ]))
  t0 <- Sys.time()
  res <- parLapplyLB(cl, tasks, run_task)
  res <- do.call(rbind, res)
  saveRDS(res, f)
  el <- as.numeric(difftime(Sys.time(), t_start, units = "mins"))
  cat(sprintf("[%s] block %d/%d (reps %d-%d) done in %.1f min; ETA %.0f min\n",
              format(Sys.time(), "%H:%M:%S"), b, nblocks, min(reps), max(reps),
              as.numeric(difftime(Sys.time(), t0, units = "mins")),
              el / i * (length(my_blocks) - i)))
}
stopCluster(cl)
cat("All replications done.  Now run:  Rscript plot_figures.R\n")
