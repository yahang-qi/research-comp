## =============================================================================
##  Scenario 5: estimated vs optimal decision boundary as the covariate dimension grows.
##  One training set (n = 800 by default) is drawn once with 50 covariates; dimension d uses
##  the first d columns (nested: 3 < 10 < 20 < 50).  R, A and the optimal rule only involve
##  X1-X3, so they are identical for every d.  Hyperparameters are chosen as in the main
##  loop (same grids, 5-fold CV).  Each panel shows the (X1, X2) plane; the other d - 2
##  covariates are averaged out by Monte Carlo:
##    shading = P(estimated rule = 1 | X1, X2),  solid line = its 0.5 contour,
##    dashed line = optimal boundary X1^2 + X2^2 = 2/3.
##  Usage (project root):  OWL_P=50 Rscript scripts/plot/plot_estimated_boundary_g5.R [n]
## =============================================================================
source("R/owl_sim_functions.R")
if (P != 50L) stop("run with OWL_P=50 (data are generated with 50 covariates)", call. = FALSE)
args <- suppressWarnings(as.integer(commandArgs(trailingOnly = TRUE)))
n <- if (length(args) >= 1 && !is.na(args[1])) args[1] else 800L
NS <- c(30L, 100L, 200L, 400L, 800L)
DIMS <- c(3L, 10L, 20L, 50L)
grid <- list(lambdas_rbf = c(0.5, 4, 32), kparam = c(2, 4, 8), lambdas_lin = c(0.5, 4, 32),
             lambdas_poly = c(0.5, 4, 32))
methods <- c("OLS", "l1-PLS", "OWL-Linear", "OWL-Gaussian", "OWL-Poly2")
col_pos <- "#FBE3D6"; col_neg <- "#DCE9F5"; col_true <- "#B71C1C"
g <- seq(-1, 1, length.out = 41); M <- 100            # plane grid, Monte Carlo draws

## ---- data: drawn once with 50 covariates, same seed as the main loop (scenario 5, replicate 1)
set.seed(1e6 * 5 + 1e3 * match(n, NS) + 1L)
dat50 <- gen_data(n, 5)
## ---- Monte Carlo design for the plane: 50 covariates, columns 1:d used for dimension d
set.seed(105)
gr <- expand.grid(a = g, b = g)
Xg50 <- matrix(runif(nrow(gr) * M * 50, -1, 1), nrow(gr) * M, 50, dimnames = list(NULL, paste0("X", 1:50)))
Xg50[, 1] <- rep(gr$a, M); Xg50[, 2] <- rep(gr$b, M)
avg <- function(d) matrix(rowMeans(matrix(d, nrow(gr), M)), length(g))

prob <- setNames(rep(list(list()), length(methods)), methods)
for (d in DIMS) {
  P <- d; XNAMES <- paste0("X", seq_len(d))                       # globals used by the fitters
  regime_fml <- as.formula(paste("~", paste(XNAMES, collapse = " + ")))
  dat <- dat50; dat$X <- dat50$X[, seq_len(d), drop = FALSE]
  Xg <- Xg50[, seq_len(d), drop = FALSE]
  set.seed(7)                                                      # same CV-fold seed for every d
  fg <- fit_owl(dat, "radial", lambdas = grid$lambdas_rbf / n, kparam = grid$kparam)
  fl <- fit_owl(dat, "linear", lambdas = grid$lambdas_lin / n)
  fp <- fit_owl(dat, "poly",   lambdas = grid$lambdas_poly / n, kparam = 2)
  b  <- fit_l1pls(dat); bo <- fit_ols(dat)
  rule <- function(f) if (is.null(f)) NULL else avg(predict_owl(f, Xg) == 1)
  prob$OLS[[as.character(d)]]            <- avg(pls_rule(bo, Xg) == 1)
  prob$`l1-PLS`[[as.character(d)]]       <- avg(pls_rule(b, Xg) == 1)
  prob$`OWL-Linear`[[as.character(d)]]   <- rule(fl)
  prob$`OWL-Gaussian`[[as.character(d)]] <- rule(fg)
  prob$`OWL-Poly2`[[as.character(d)]]    <- rule(fp)
  cat("d =", d, "done; selected (Gaussian, Linear, Poly2):", sel_str(fg), sel_str(fl), sel_str(fp), "\n")
}

## ---- optimal boundary on a fine grid
gt <- seq(-1, 1, length.out = 401)
ztrue <- outer(gt, gt, function(a, b) a^2 + b^2 - 2/3)

pal <- colorRampPalette(c(col_neg, col_pos))(50)
cont <- function(z, x = g, ...) if (diff(range(z)) > 0 && min(z) < 0.5 && max(z) > 0.5)
  contour(x, x, z, levels = 0.5, drawlabels = FALSE, add = TRUE, ...)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
pdf(if (n == 800L) "outputs/figures/Estimated_boundaries_g5.pdf" else sprintf("outputs/figures/Estimated_boundaries_g5_n%d.pdf", n),
    width = 11, height = 14)
## top strip = stand-alone legend, below it the methods x dimensions panels
nr <- length(methods); nc <- length(DIMS)
layout(rbind(1, matrix(1 + seq_len(nr * nc), nr, nc, byrow = TRUE)), heights = c(0.45, rep(2.7, nr)))
par(mar = c(0, 0, 0, 0)); plot.new()
legend("center", horiz = TRUE, legend = c("P(D = 1) = 1", "P(D = 1) = 0", "estimated boundary", "optimal boundary"),
       fill = c(col_pos, col_neg, NA, NA), border = c("black", "black", NA, NA),
       lty = c(NA, NA, 1, 2), lwd = c(NA, NA, 2, 1.6), col = c(NA, NA, "black", col_true),
       bty = "n", cex = 1.3, x.intersp = 0.6, text.width = c(0.16, 0.16, 0.2, 0.2) * 1.0, merge = FALSE)
par(mar = c(4.2, 4.2, 2.6, 1))
for (m in methods) for (d in DIMS) {
  z <- prob[[m]][[as.character(d)]]
  plot(NA, xlim = c(-1, 1), ylim = c(-1, 1), las = 1, asp = 1,
       xlab = expression(X[1]), ylab = expression(X[2]), main = paste0(m, ", d = ", d), cex.main = 1.1)
  if (!is.null(z)) { image(g, g, z, col = pal, zlim = c(0, 1), add = TRUE, useRaster = TRUE); cont(z, lwd = 2) }
  contour(gt, gt, ztrue, levels = 0, drawlabels = FALSE, add = TRUE, lwd = 1.6, lty = 2, col = col_true)
  box()
}
dev.off()
