## =============================================================================
##  Draw Figures 2 & 3 of Zhao et al. (2012) from the checkpointed results.
##    Figure 2: MSE of value functions      = mean_r { V(d_opt) - V(d_hat_r) }^2
##    Figure 3: MSE of misclassification     = mean_r { P(d_hat_r != d_opt) }^2
##  (both evaluated on the 10,000-subject validation set of each scenario)
## =============================================================================
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
## OUT_DIR may be a comma-separated list of result folders (e.g. outputs/results/g5,outputs/results/g5_poly)
RES_DIR <- strsplit(Sys.getenv("OUT_DIR", "outputs/results/ex1-4"), ",")[[1]]; TAG <- Sys.getenv("FIG_TAG", "")
files <- list.files(RES_DIR, pattern = "\\.rds$", full.names = TRUE)
if (!length(files))
  stop("No results yet: run  Rscript scripts/run/run_simulation.R  first (and let at least one block finish).",
       call. = FALSE)
cols <- c("method", "value", "vopt", "mis", "rep", "scenario", "n")   # shared by all result types
res <- do.call(rbind, lapply(files, function(f) readRDS(f)[, cols]))
methods <- intersect(c("OLS", "l1-PLS", "OWL-Linear", "OWL-Gaussian", "OWL-Poly2"), unique(res$method))

summ <- do.call(rbind, lapply(split(res, list(res$scenario, res$n, res$method), drop = TRUE),
  function(d) data.frame(scenario = d$scenario[1], n = d$n[1], method = d$method[1],
                         reps = sum(!is.na(d$value)),
                         mse_value = mean((d$vopt - d$value)^2, na.rm = TRUE),
                         mse_mis   = mean(d$mis^2, na.rm = TRUE),
                         mean_value_loss = mean(d$vopt - d$value, na.rm = TRUE),
                         mean_mis  = mean(d$mis, na.rm = TRUE))))
summ <- summ[order(summ$scenario, summ$method, summ$n), ]

sty <- data.frame(method = c("OLS", "l1-PLS", "OWL-Linear", "OWL-Gaussian", "OWL-Poly2"),
                  lty = c(3, 2, 4, 1, 1), lwd = c(1.6, 1.6, 1.6, 1.8, 1.8),
                  col = c("grey20", "#2E7D32", "#D84315", "black", "#1565C0"))
sty <- sty[match(methods, sty$method), ]
leg <- as.expression(lapply(methods, function(m) switch(m, "l1-PLS" = quote(l[1]*"-PLS"),
                                                        "OWL-Poly2" = quote("OWL-Poly"[2]), m)))

draw <- function(var, file_stub) {
  pdf(paste0("outputs/figures/", file_stub, TAG, ".pdf"), width = 8, height = 7)
  scens <- sort(unique(summ$scenario))
  par(mfrow = if (length(scens) > 1) c(2, 2) else c(1, 1), mar = c(4.2, 4.2, 2.6, 1))
  for (s in scens) {
    d <- summ[summ$scenario == s, ]
    plot(NA, xlim = c(0, 800), ylim = c(0, max(d[[var]], na.rm = TRUE) * 1.05),
         xlab = "n", ylab = "MSE", main = paste("Example", s), las = 1,
         cex.main = 1.3)
    for (m in methods) {
      dm <- d[d$method == m, ]; st <- sty[sty$method == m, ]
      lines(dm$n, dm[[var]], lty = st$lty, lwd = st$lwd, col = st$col)
    }
    legend("topright", legend = leg, lty = sty$lty, lwd = sty$lwd, col = sty$col,
           bty = "n", cex = 0.85, seg.len = 2.5)
  }
  dev.off()
}
draw("mse_value", "Figure2_MSE_value")
draw("mse_mis",   "Figure3_MSE_misclassification")
print(summ, digits = 3, row.names = FALSE)
