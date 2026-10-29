## =============================================================================
##  Estimated decision boundaries of OLS, l1-PLS, OWL-Linear and OWL-Gaussian (one
##  training set per scenario = replicate 1 of the main simulation: same seed, same fitting
##  order and same CV tuning grids as run_simulation.R / one_rep()).
##  Usage (from the project root):  Rscript scripts/plot/plot_estimated_boundary.R [n]      (default n = 800)
##  The fitted rules use all 50 covariates, so each panel shows the plane of the
##  covariates that matter (X1, X2 for Examples 1-3; X7, X8 for Example 4) and
##  averages the rule over the remaining covariates ~ U(-1, 1) by Monte Carlo:
##    shading = P(estimated rule = 1 | plane coordinates),  solid line = its 0.5 contour,
##    dashed line = same contour for the true rule D*(X) = sign(delta(X)).
## =============================================================================
source("R/owl_sim_functions.R")
args <- suppressWarnings(as.integer(commandArgs(trailingOnly = TRUE)))
n <- if (length(args) >= 1 && !is.na(args[1])) args[1] else 800L
NS <- c(30L, 100L, 200L, 400L, 800L)
grid <- list(lambdas_rbf = c(0.5, 4, 32), kparam = c(2, 4, 8), lambdas_lin = c(0.5, 4, 32))

col_pos <- "#FBE3D6"; col_neg <- "#DCE9F5"
g <- seq(-1, 1, length.out = 41); M <- 100          # plane grid, Monte Carlo draws
axes <- list(c(1, 2), c(1, 2), c(1, 2), c(7, 8))
methods <- c("OLS", "l1-PLS", "OWL-Linear", "OWL-Gaussian")

prob <- setNames(rep(list(list()), length(methods)), methods); truth <- list()

## true rule on a finer grid (Monte Carlo average over the other covariates, Example 4)
gt <- seq(-1, 1, length.out = 201)
for (s in 1:4) {
  ax <- axes[[s]]; set.seed(200 + s); Mt <- if (s == 4) 4000 else 1
  truth[[s]] <- sapply(gt, function(b) {            # one column of the plane per call
    X <- matrix(0, length(gt) * Mt, P)             # delta only involves X1, X3, X5-X8
    for (j in setdiff(c(1, 3, 5, 6), ax)) X[, j] <- runif(nrow(X), -1, 1)
    X[, ax[1]] <- rep(gt, Mt); X[, ax[2]] <- b
    rowMeans(matrix(delta_fun[[as.character(s)]](X) > 0, length(gt), Mt))
  })
}
for (s in 1:4) {
  set.seed(1e6 * s + 1e3 * match(n, NS) + 1L)
  dat <- gen_data(n, s)
  fg <- fit_owl(dat, "radial", lambdas = grid$lambdas_rbf / n, kparam = grid$kparam)  # same order
  fl <- fit_owl(dat, "linear", lambdas = grid$lambdas_lin / n)                        # as one_rep()
  b  <- fit_l1pls(dat)
  bo <- fit_ols(dat)
  ax <- axes[[s]]
  set.seed(100 + s)
  gr <- expand.grid(a = g, b = g)
  Xg <- matrix(runif(nrow(gr) * M * P, -1, 1), nrow(gr) * M, P, dimnames = list(NULL, XNAMES))
  Xg[, ax[1]] <- rep(gr$a, M); Xg[, ax[2]] <- rep(gr$b, M)
  avg <- function(d) matrix(rowMeans(matrix(d, nrow(gr), M)), length(g))
  prob$OLS[[s]]            <- avg(pls_rule(bo, Xg) == 1)
  prob$`l1-PLS`[[s]]       <- avg(pls_rule(b, Xg) == 1)
  prob$`OWL-Linear`[[s]]   <- avg(predict_owl(fl, Xg) == 1)
  prob$`OWL-Gaussian`[[s]] <- avg(predict_owl(fg, Xg) == 1)
  cat("scenario", s, "done\n")
}

cont <- function(z, x = g, ...) if (diff(range(z)) > 0 && min(z) < 0.5 && max(z) > 0.5)
  contour(x, x, z, levels = 0.5, drawlabels = FALSE, add = TRUE, ...)
pal <- colorRampPalette(c(col_neg, col_pos))(50)

dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
pdf("outputs/figures/Estimated_boundaries.pdf", width = 12, height = 12.2)
par(mfrow = c(4, 4), mar = c(4.2, 4.2, 2.6, 1))
for (m in methods) for (s in 1:4) {
  ax <- axes[[s]]
  plot(NA, xlim = c(-1, 1), ylim = c(-1, 1), las = 1, asp = 1,
       xlab = bquote(X[.(ax[1])]), ylab = bquote(X[.(ax[2])]),
       main = paste0(m, ", Example ", s), cex.main = 1.1)
  image(g, g, prob[[m]][[s]], col = pal, zlim = c(0, 1), add = TRUE, useRaster = TRUE)
  cont(prob[[m]][[s]], lwd = 2)
  cont(truth[[s]], gt, lwd = 1.6, lty = 2, col = "#B71C1C")
  box()
  if (s == 1 && m == methods[1])
    legend("bottomleft", legend = c("P(D = 1) = 1", "P(D = 1) = 0", "estimated boundary", "true boundary"),
           fill = c(col_pos, col_neg, NA, NA), border = c("black", "black", NA, NA),
           lty = c(NA, NA, 1, 2), lwd = c(NA, NA, 2, 1.6), col = c(NA, NA, "black", "#B71C1C"),
           bty = "n", cex = 0.7, x.intersp = 0.5, merge = FALSE)
}
dev.off()
