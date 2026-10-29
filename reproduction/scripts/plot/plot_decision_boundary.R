## =============================================================================
##  Decision boundary D*(X) = sign(delta(X)) for the four simulation scenarios
##  (delta_fun from R/owl_sim_functions.R, copied here so no packages are needed).
##  Examples 1-3 depend on (X1, X2) only, so the boundary delta = 0 is drawn there.
##  Example 4 depends on X1, X3, X5, X6, X7, X8: it is drawn in the (X7, X8) plane
##  with the other covariates averaged out by Monte Carlo (X ~ U(-1, 1)^p);
##  shading = P(D* = 1 | X7, X8), line = the 0.5 contour.
## =============================================================================
delta_fun <- list(
  function(X) 0.442 * (1 - X[, 1] - X[, 2]),
  function(X) X[, 2] - 0.25 * X[, 1]^2 - 1,
  function(X) (0.5 - X[, 1]^2 - X[, 2]^2) * (X[, 1]^2 + X[, 2]^2 - 0.3),
  function(X) 1 - X[, 1]^3 + exp(X[, 3]^2 + X[, 5]) + 0.6 * X[, 6] - (X[, 7] + X[, 8])^2)

col_pos <- "#FBE3D6"; col_neg <- "#DCE9F5"       # D* = 1 / D* = -1
g <- seq(-1, 1, length.out = 201)
set.seed(1)

dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
cairo_pdf("outputs/figures/Decision_boundaries.pdf", width = 8, height = 7)   # cairo for the blackboard-bold P
par(mfrow = c(2, 2), mar = c(4.5, 4.8, 2.8, 1), cex = 1, cex.axis = 0.95, cex.lab = 1.15)
for (s in 1:3) {
  grid <- expand.grid(x1 = g, x2 = g)
  z <- matrix(delta_fun[[s]](cbind(grid$x1, grid$x2)), length(g))
  plot(NA, xlim = c(-1, 1), ylim = c(-1, 1), xlab = expression(X[1]), ylab = expression(X[2]),
       main = paste("Example", s), las = 1, cex.main = 1.25, asp = 1, xaxt = "n")
  axis(1, at = seq(-1, 1, 0.5))
  image(g, g, ifelse(z > 0, 1, 0), col = c(col_neg, col_pos), breaks = c(-.5, .5, 1.5), add = TRUE, useRaster = TRUE)
  contour(g, g, z, levels = 0, drawlabels = FALSE, add = TRUE, lwd = 2)
  box()
  legend("bottomleft", legend = c("D* = 1", "D* = -1"),
         fill = c(col_pos, col_neg), bty = "n", cex = 0.95)
}

## Example 4: Monte Carlo over the remaining covariates
g4 <- seq(-1, 1, length.out = 201); M <- 2000
others <- matrix(runif(M * 8, -1, 1), M, 8)       # columns -> X1..X6 used; X7, X8 overwritten
p <- outer(g4, g4, Vectorize(function(a, b) {
  X <- others; X[, 7] <- a; X[, 8] <- b
  mean(delta_fun[[4]](X) > 0)
}))
plot(NA, xlim = c(-1, 1), ylim = c(-1, 1), xlab = expression(X[7]), ylab = expression(X[8]),
     main = "Example 4", las = 1, cex.main = 1.25, asp = 1, xaxt = "n")
axis(1, at = seq(-1, 1, 0.5))
image(g4, g4, p, col = colorRampPalette(c(col_neg, col_pos))(50), zlim = c(0, 1), add = TRUE, useRaster = TRUE)
contour(g4, g4, p, levels = 0.5, drawlabels = FALSE, add = TRUE, lwd = 2)
box()
legend("topleft", legend = c("\u2119(D* = 1) = 1", "\u2119(D* = 1) = 0"),
       fill = c(col_pos, col_neg), bty = "n", cex = 0.95)
dev.off()
