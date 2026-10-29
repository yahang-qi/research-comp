## =============================================================================
##  Example 5 only: Figure 2 (MSE of policy value) and Figure 3 (MSE of
##  misclassification) merged into one two-panel figure with a shared legend.
##  Reads the raw results of outputs/results/g5/ and outputs/results/g5_poly/ (1000 reps).
##      Rscript scripts/plot/plot_figures_g5.R   ->  outputs/figures/Figure2_3_g5.pdf
##  (scripts/plot/plot_figures.R is left untouched: it reproduces the paper's figures.)
## =============================================================================
RES_DIR <- strsplit(Sys.getenv("OUT_DIR", "outputs/results/g5,outputs/results/g5_poly"), ",")[[1]]
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
files <- list.files(RES_DIR, pattern = "\\.rds$", full.names = TRUE)
if (!length(files)) stop("No results found in ", paste(RES_DIR, collapse = ", "), call. = FALSE)
cols <- c("method", "value", "vopt", "mis", "rep", "scenario", "n")
res <- do.call(rbind, lapply(files, function(f) readRDS(f)[, cols]))
res <- res[res$scenario == 5, ]
methods <- intersect(c("OLS", "l1-PLS", "OWL-Linear", "OWL-Gaussian", "OWL-Poly2"), unique(res$method))

summ <- do.call(rbind, lapply(split(res, list(res$n, res$method), drop = TRUE),
  function(d) data.frame(n = d$n[1], method = d$method[1],
                         mse_value = mean((d$vopt - d$value)^2, na.rm = TRUE),
                         mse_mis   = mean(d$mis^2, na.rm = TRUE))))

sty <- data.frame(method = c("OLS", "l1-PLS", "OWL-Linear", "OWL-Gaussian", "OWL-Poly2"),
                  lty = c(3, 2, 4, 1, 1), lwd = c(1.6, 1.6, 1.6, 1.8, 1.8),
                  col = c("grey20", "#2E7D32", "#D84315", "black", "#1565C0"))
sty <- sty[match(methods, sty$method), ]
leg <- as.expression(lapply(methods, function(m) switch(m, "l1-PLS" = quote(l[1]*"-PLS"),
                                                        "OWL-Poly2" = quote("OWL-Poly"[2]), m)))

panel <- function(var, ylab) {
  plot(NA, xlim = c(0, 800), ylim = c(0, max(summ[[var]], na.rm = TRUE) * 1.05),
       xlab = "n", ylab = "", las = 1)
  mtext(ylab, side = 2, line = 3.9, cex = 1.35)
  for (m in methods) {
    dm <- summ[summ$method == m, ]; dm <- dm[order(dm$n), ]; st <- sty[sty$method == m, ]
    lines(dm$n, dm[[var]], lty = st$lty, lwd = st$lwd + 1, col = st$col)
  }
}

## geometry (inches): each axes box has the same height/width ratio as the panels of Figures 2 & 3
## in plot_figures.R (8x7 in, 2x2 layout, mar = c(4.2, 4.2, 2.6, 1) -> 3.137 x 2.371 in)
ASPECT <- 2.3712 / 3.1368
W <- 12; panel_w <- W / 2
mai_l <- 1.05; mai_r <- 0.2; mai_b <- 0.85; mai_t <- 0.1
box_w <- panel_w - mai_l - mai_r; box_h <- ASPECT * box_w
panel_h <- box_h + mai_b + mai_t; leg_h <- 0.5
pdf("outputs/figures/Figure2_3_g5.pdf", width = W, height = panel_h + leg_h)
layout(matrix(c(1, 2, 3, 3), 2, byrow = TRUE), heights = c(panel_h, leg_h))
par(mai = c(mai_b, mai_l, mai_t, mai_r), mgp = c(3.2, 1, 0), cex.axis = 1.3, cex.lab = 1.4)
panel("mse_value", "MSE of policy value")
panel("mse_mis",   "MSE of misclassification")
## shared legend: its own row below both panels
par(mai = c(0, 0, 0, 0))
plot.new()
legend("center", legend = leg, lty = sty$lty, lwd = sty$lwd + 1, col = sty$col,
       horiz = TRUE, bty = "n", cex = 1.5, seg.len = 3, x.intersp = 0.8)
dev.off()
