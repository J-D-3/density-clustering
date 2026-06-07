#!/usr/bin/env Rscript
# Run mhahsler/dbscan's OPTICS on a coordinates CSV and emit predicted labels, so the
# quality/timing harness can score it next to this library and scikit-learn.
#
# It times only optics() (the ordering), matching what our optics_quality_compare times;
# extractXi (the cheap extraction) runs untimed. dbscan's noise label 0 is remapped to -1
# to match our convention. The exact reachability ordering will differ from ours on ties
# (dbscan/ELKI break ties high-index-first, we low-index-first) -- compare clusters
# (ARI/NMI/Rand) and timing, not bit-identical orderings.
#
# Usage:  Rscript run_dbscan_r.R coords.csv out_labels.csv [minPts=10] [xi=0.05] [eps=auto]
# stdout: <out_labels.csv> gets "point_index,label"; stderr: TIMING n=.. dim=.. dbscan_ms=..

suppressMessages(library(dbscan))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) { cat("usage: run_dbscan_r.R coords.csv out.csv [minPts] [xi] [eps]\n", file = stderr()); quit(status = 2) }
coords_path <- args[1]
out_path    <- args[2]
minPts <- if (length(args) >= 3) as.integer(args[3]) else 10L
xi     <- if (length(args) >= 4) as.numeric(args[4]) else 0.05
eps    <- if (length(args) >= 5) as.numeric(args[5]) else -1.0

df <- read.csv(coords_path)
xcols <- grep("^x", names(df))
if (length(xcols) == 0) xcols <- seq_len(ncol(df))
X <- as.matrix(df[, xcols, drop = FALSE])
n <- nrow(X); d <- ncol(X)

# eps is OPTICS's generating distance; a large value yields the full structure (the
# FOPTICS paper used eps = infinity). Default to well beyond the data extent.
if (eps <= 0) {
  rng <- apply(X, 2, function(col) max(col) - min(col))
  eps <- 10.0 * max(rng) + 1.0
}

t0 <- Sys.time()
res <- optics(X, eps = eps, minPts = minPts)
ms <- ceiling(as.numeric(difftime(Sys.time(), t0, units = "secs")) * 1000.0)

lab <- tryCatch({
  xi_res <- extractXi(res, xi = xi)
  xi_res$cluster
}, error = function(e) rep(0L, n))
lab[lab == 0] <- -1L  # dbscan noise (0) -> our convention (-1)

out <- data.frame(point_index = 0:(n - 1), label = as.integer(lab))
write.csv(out, out_path, row.names = FALSE, quote = FALSE)
cat(sprintf("TIMING n=%d dim=%d dbscan_ms=%d\n", n, d, as.integer(ms)), file = stderr())
