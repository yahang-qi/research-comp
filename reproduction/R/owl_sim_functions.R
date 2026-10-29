## =============================================================================
##  Reproduction of Figures 2 & 3 in
##  Zhao, Zeng, Rush & Kosorok (2012). Estimating Individualized Treatment Rules
##  Using Outcome Weighted Learning. JASA 107(499), 1106-1118.
##
##  Core functions: data generation, the four competing methods, and metrics.
##  OWL (Gaussian and linear kernel) is fitted with DynTxRegime::owl(); fit_owl() also
##  accepts kernel = "poly" ((1 + x'x')^kparam, used by run_poly.R).
##  The number of covariates P is read from the environment variable OWL_P (default 50).
## =============================================================================

suppressPackageStartupMessages({
  library(DynTxRegime)   # owl(), buildModelObj() (via modelObj), optTx()
  library(glmnet)        # l1-PLS (Qian & Murphy 2011)
})

## -----------------------------------------------------------------------------
## Run-time patch for DynTxRegime::owl() with a non-linear (e.g. radial) kernel
## and the hinge surrogate (checked against DynTxRegime 4.16).
##
## For the hinge loss, owl() solves the weighted-SVM dual with kernlab::ipop().
## The decision function is f(x) = sum_i alpha_i * y_i * K(x_i, x) + b with
## y_i = sign(A_i * R_i).  For the linear kernel the package correctly forms
## (alpha * y) %*% X, but for the radial kernel it stores the raw dual
## variables alpha_i (all >= 0) as the kernel coefficients, i.e. it drops y_i.
## Every training point then pushes f(x) upwards, so the estimated rule is
## (almost) always "treat everyone with A = 1", and cross-validation is
## computed from these wrong rules too.  The patch multiplies by y_i.
##
## Second fix, for the linear kernel: owl() passes ipop() the full n x n dual
## matrix H = (y y') * (X X'), whose rank is at most p = 50.  For n = 800 the
## interior-point solve occasionally fails with "system is computationally
## singular".  ipop() also accepts an n x m factor Z with Z Z' = H (n > m) and
## then uses a Sherman-Morrison-Woodbury solve; Z = diag(y) X gives the same QP.
## (On that path ipop returns primal as an n x 1 matrix, so it is flattened.)
## For n <= m (e.g. n = 30 < p) ipop would misread Z, so the full H is kept.
## -----------------------------------------------------------------------------
patch_owl_kernel_coef <- function() {
  ns <- asNamespace("DynTxRegime")
  subs <- list(
    c(old = 'res[["par"]] <- c(-optimResults@dual, optimResults@primal)',
      new = 'res[["par"]] <- c(-optimResults@dual, optimResults@primal * methodObject@txSignR)'),
    c(old = 'argList[["H"]] <- h1 * {\n methodObject@x %*% t(methodObject@x)\n }',
      new = paste('argList[["H"]] <- if (nrow(methodObject@x) > ncol(methodObject@x))',
                  'methodObject@txSignR * methodObject@x else',
                  'h1 * tcrossprod(methodObject@x)')),
    c(old = 'if (optimResults@how != "converged") {',
      new = 'optimResults@primal <- as.vector(optimResults@primal)\n if (optimResults@how != "converged") {'))
  norm <- function(f) paste(trimws(deparse(f)), collapse = "\n ")
  m <- getMethod(".optimFunc", ".owl", where = ns)
  if (!all(vapply(subs, function(s) grepl(s[["old"]], norm(m@.Data), fixed = TRUE),
                  logical(1)))) {
    message("DynTxRegime .optimFunc(.owl) differs from 4.16; patch not applied.")
    return(invisible(FALSE))
  }
  suppressMessages(trace(".optimFunc", signature = ".owl", where = ns, print = FALSE,
        edit = eval(bquote(function(name, file, title) {
          f <- name
          txt <- .(norm)(body(f))
          for (s in .(subs)) txt <- gsub(s[["old"]], s[["new"]], txt, fixed = TRUE)
          body(f) <- str2lang(txt)
          f
        }))))
  invisible(TRUE)
}
patch_owl_kernel_coef()

P <- as.integer(Sys.getenv("OWL_P", "50"))                                  # number of prognostic variables
XNAMES <- paste0("X", seq_len(P))

## ---- Section 4: interaction terms T0(X, A) = delta(X) * A --------------------
## delta(X) is the treatment contrast / 2;  d_opt(X) = sign(delta(X)).
delta_fun <- list(
  `1` = function(X) 0.442 * (1 - X[, 1] - X[, 2]),
  `2` = function(X) X[, 2] - 0.25 * X[, 1]^2 - 1,
  `3` = function(X) (0.5 - X[, 1]^2 - X[, 2]^2) * (X[, 1]^2 + X[, 2]^2 - 0.3),
  `4` = function(X) 1 - X[, 1]^3 + exp(X[, 3]^2 + X[, 5]) + 0.6 * X[, 6] -
                    (X[, 7] + X[, 8])^2,
  `5` = function(X) 3 * (X[, 1]^2 + X[, 2]^2 - 2/3)
)

## R ~ N(Q0, 1),  Q0 = 1 + 2 X1 + X2 + 0.5 X3 + T0(X, A),  A in {-1, 1}, P(A=1)=1/2
gen_data <- function(n, scenario) {
  X <- matrix(runif(n * P, -1, 1), n, P, dimnames = list(NULL, XNAMES))
  A <- sample(c(-1L, 1L), n, replace = TRUE)
  delta <- delta_fun[[as.character(scenario)]](X)
  R <- 1 + 2 * X[, 1] + X[, 2] + 0.5 * X[, 3] + delta * A + rnorm(n)
  list(X = X, A = A, R = R, dopt = ifelse(delta > 0, 1L, -1L))
}

## ---- Evaluation criteria -------------------------------------------------------
## Value of rule D on validation data (Murphy et al. 2001), P(A) = 1/2:
##   V(D) = Pn[ I(A = D(X)) R / P(A) ] / Pn[ I(A = D(X)) / P(A) ]
ipw_value <- function(d, A, R) { w <- (A == d) / 0.5; sum(w * R) / sum(w) }

## ---- Method 1 & 2: OWL via DynTxRegime::owl ----------------------------------
## Constant propensity model; P(A = 1) is estimated as the sample proportion.
moPropen <- buildModelObj(model = ~1,
                          solver.method = "glm",
                          solver.args = list(family = "binomial"),
                          predict.method = "predict.glm",
                          predict.args = list(type = "response"))
regime_fml <- as.formula(paste("~", paste(XNAMES, collapse = " + ")))

## Notes on parametrisation in DynTxRegime::owl (hinge surrogate):
##  * The dual box constraint is 0 <= alpha_i <= lambda * |R_i| / pi_i, so the
##    'lambdas' argument plays the role of 1 / (n * lambda_n) in the paper.
##  * kernel = "radial": k(x, x') = exp(-||x - x'||^2 / (2 kparam^2)),
##    i.e. kparam = 1 / (sqrt(2) * sigma_n) in the paper's notation.
##  * Rewards are shifted to be non-negative (R - min R), as in Zhao et al. (2012).
##  * cvFolds = 5: (lambda, kparam) chosen to maximise the cross-validated value.
fit_owl <- function(dat, kernel = c("radial", "linear", "poly"), lambdas, kparam = NULL,
                    cvFolds = 5L) {
  kernel <- match.arg(kernel)
  df <- data.frame(dat$X, A = dat$A)
  rew <- dat$R - min(dat$R)
  fit <- NULL
  err <- NULL
  invisible(capture.output(suppressWarnings(suppressMessages(
    fit <- tryCatch(
      owl(moPropen = moPropen, data = df, reward = rew, txName = "A",
          regime = regime_fml, lambdas = lambdas, cvFolds = cvFolds,
          kernel = kernel, kparam = if (kernel != "linear") kparam else NULL,
          surrogate = "hinge", verbose = 0L),
      error = function(e) { err <<- conditionMessage(e); NULL })
  ))))
  ## a failed fit is recorded as NA for this method only (see one_rep)
  if (!is.null(err)) warning(sprintf("OWL-%s fit failed (n = %d): %s",
                                     kernel, nrow(df), err), call. = FALSE)
  fit
}

predict_owl <- function(fit, Xnew) {
  as.integer(optTx(fit, newdata = data.frame(Xnew))$optimalTx)
}

## ---- Method 3: l1-PLS (Qian & Murphy 2011) -----------------------------------
## Basis (1, X, A, XA) with LASSO; lambda chosen by 5-fold CV to maximise the
## IPW value of the induced rule on the held-out folds ("follows similarly").
pls_design <- function(X, A) cbind(X, A = A, X * A)
pls_rule <- function(beta, X) {         # sign of E(R|X,1) - E(R|X,-1)
  bA <- beta["A"]; bXA <- beta[(P + 2):(2 * P + 1)]
  ifelse(drop(bA + X %*% bXA) > 0, 1L, -1L)
}
fit_l1pls <- function(dat, nfolds = 5L) {
  Z <- pls_design(dat$X, dat$A); n <- nrow(Z)
  full <- glmnet(Z, dat$R, alpha = 1, nlambda = 50)
  lam <- full$lambda
  folds <- sample(rep_len(seq_len(nfolds), n))
  cvval <- matrix(NA_real_, nfolds, length(lam))
  for (k in seq_len(nfolds)) {
    tr <- folds != k; te <- !tr
    fk <- glmnet(Z[tr, , drop = FALSE], dat$R[tr], alpha = 1, lambda = lam)
    B <- as.matrix(coef(fk))[-1, , drop = FALSE]
    rownames(B) <- colnames(Z)
    for (j in seq_along(lam)) {
      d <- pls_rule(B[, j], dat$X[te, , drop = FALSE])
      cvval[k, j] <- ipw_value(d, dat$A[te], dat$R[te])
    }
  }
  m <- colMeans(cvval, na.rm = TRUE)
  j <- which(m == max(m))[1]            # ties: largest lambda (sparsest rule)
  beta <- as.matrix(coef(full))[-1, j]; names(beta) <- colnames(Z)
  beta
}

## ---- Method 4: OLS on (1, X, A, XA), no variable selection --------------------
## The design has 2p + 2 columns (102 for p = 50), so for n <= 2p + 2 the least-squares
## problem is under-determined; we use the minimum-norm least-squares solution
## (Moore-Penrose pseudo-inverse), which equals ordinary OLS when n > 2p + 2.
fit_ols <- function(dat) {
  Z <- cbind(1, pls_design(dat$X, dat$A))
  s <- svd(Z)
  tol <- max(dim(Z)) * max(s$d) * .Machine$double.eps
  pos <- s$d > tol
  beta <- drop(s$v[, pos, drop = FALSE] %*% (crossprod(s$u[, pos, drop = FALSE], dat$R) / s$d[pos]))
  names(beta) <- c("(Intercept)", colnames(pls_design(dat$X[1:2, ], dat$A[1:2])))
  beta[-1]
}

## ---- One replicate: fit all four methods, evaluate on validation data -------
one_rep <- function(n, scenario, val, grid) {
  dat <- gen_data(n, scenario)
  rules <- list()
  fg <- fit_owl(dat, "radial", lambdas = grid$lambdas_rbf / n, kparam = grid$kparam)
  rules$`OWL-Gaussian` <- if (is.null(fg)) NA else predict_owl(fg, val$X)
  fl <- fit_owl(dat, "linear", lambdas = grid$lambdas_lin / n)
  rules$`OWL-Linear` <- if (is.null(fl)) NA else predict_owl(fl, val$X)
  rules$`l1-PLS` <- pls_rule(fit_l1pls(dat), val$X)
  rules$OLS <- pls_rule(fit_ols(dat), val$X)

  vopt <- ipw_value(val$dopt, val$A, val$R)
  out <- lapply(names(rules), function(m) {
    d <- rules[[m]]
    if (anyNA(d)) return(data.frame(method = m, value = NA, vopt = vopt, mis = NA))
    data.frame(method = m, value = ipw_value(d, val$A, val$R), vopt = vopt,
               mis = mean(d != val$dopt))
  })
  res <- do.call(rbind, out)
  res$selected <- c(
    sel_str(fg),
    sel_str(fl),
    NA, NA)
  res
}

## (lambda[, kparam]) selected by DynTxRegime's cross-validation, for diagnostics
sel_str <- function(fit) {
  if (is.null(fit)) return(NA_character_)
  s <- summary(fit)$optim
  k <- tryCatch(fit@analysis@optim@kernel@kparam, error = function(e) NULL)
  paste(c(signif(s$lambda, 3), if (length(k)) signif(k, 3)), collapse = "/")
}
