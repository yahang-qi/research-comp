## OWL with a degree-2 polynomial kernel k(x,x') = (1 + x'x)^2 on scenarios 5 and 6.
## Same replications / seeds / validation sets / lambda grid as run_simulation.R.
##   Rscript scripts/run/run_poly.R 100 16
library(parallel)
args <- as.integer(commandArgs(trailingOnly = TRUE))
REPS <- args[1]; CORES <- args[2]
NS <- c(30L, 100L, 200L, 400L, 800L); NVAL <- 10000L; SCEN <- as.integer(strsplit(Sys.getenv("SCENARIOS", "5"), ",")[[1]]); OUT <- Sys.getenv("OUT_DIR", "outputs/results/g5_poly")
grid <- list(lambdas_poly = c(0.5, 4, 32), kparam = 2)
source("R/owl_sim_functions.R")
vals <- lapply(SCEN, function(s) { set.seed(20120000 + s); gen_data(NVAL, s) }); names(vals) <- SCEN
cl <- makeCluster(CORES)
clusterEvalQ(cl, { Sys.setenv(OPENBLAS_NUM_THREADS = "1", OMP_NUM_THREADS = "1"); source("R/owl_sim_functions.R"); NULL })
clusterExport(cl, c("vals", "grid", "NS"))
run_task <- function(task) {
  s <- task$s; n <- task$n; r <- task$r; val <- vals[[as.character(s)]]
  set.seed(1e6 * s + 1e3 * match(n, NS) + r)
  out <- tryCatch({
    dat <- gen_data(n, s)
    f <- fit_owl(dat, "poly", lambdas = grid$lambdas_poly / n, kparam = grid$kparam)
    vopt <- ipw_value(val$dopt, val$A, val$R)
    if (is.null(f)) data.frame(method = "OWL-Poly2", value = NA, vopt = vopt, mis = NA)
    else { d <- predict_owl(f, val$X)
      data.frame(method = "OWL-Poly2", value = ipw_value(d, val$A, val$R), vopt = vopt, mis = mean(d != val$dopt)) }
  }, error = function(e) NULL)
  if (!is.null(out)) { out$rep <- r; out$scenario <- s; out$n <- n }
  out
}
tasks <- expand.grid(r = seq_len(REPS), s = SCEN, n = NS); tasks <- tasks[order(-tasks$n), ]
tasks <- lapply(seq_len(nrow(tasks)), function(i) as.list(tasks[i, ]))
res <- do.call(rbind, parLapplyLB(cl, tasks, run_task))
dir.create(OUT, recursive = TRUE, showWarnings = FALSE); saveRDS(res, file.path(OUT, "poly2.rds"))
stopCluster(cl)
