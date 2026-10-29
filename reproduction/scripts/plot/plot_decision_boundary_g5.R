## Decision boundary D*(X) = sign(delta(X)) for g5: delta = 3(X1^2 + X2^2 - 2/3),
## depends on (X1, X2) only; boundary is the circle X1^2 + X2^2 = 2/3.
col_pos <- "#FBE3D6"; col_neg <- "#DCE9F5"       # D* = 1 / D* = -1
g <- seq(-1, 1, length.out = 401)
z <- outer(g, g, function(a, b) 3 * (a^2 + b^2 - 2/3))
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
pdf("outputs/figures/Decision_boundary_g5.pdf", width = 5, height = 5)
par(mar = c(4.2, 4.2, 2.6, 1))
plot(NA, xlim = c(-1, 1), ylim = c(-1, 1), xlab = expression(X[1]), ylab = expression(X[2]),
     main = "Example 5", las = 1, cex.main = 1.3, asp = 1)
image(g, g, ifelse(z > 0, 1, 0), col = c(col_neg, col_pos), breaks = c(-.5, .5, 1.5), add = TRUE, useRaster = TRUE)
contour(g, g, z, levels = 0, drawlabels = FALSE, add = TRUE, lwd = 2)
box()
legend("bottomleft", legend = c("D* = 1", "D* = -1"), fill = c(col_pos, col_neg), bty = "n", cex = 0.85)
dev.off()
