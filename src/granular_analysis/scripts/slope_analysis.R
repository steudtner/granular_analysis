## ============================================================
## Daten einlesen und bereinigen (ohne readr)
## ============================================================
args <- commandArgs(trailingOnly = TRUE)
csv_path <- args[1]
out_dir <- args[2]

# Lade segmented, falls vorhanden
if (!requireNamespace("segmented", quietly = TRUE)) {
  stop("R package 'segmented' is required but not installed.")
}
library(segmented)
set.seed(20260924) # Reproducible bootstrap restart in future fits

# Accept both the current header-bearing CSV and older headerless files.
df <- read.csv(csv_path, header=TRUE)
if (!all(c("x", "z_max") %in% names(df))) {
  df <- read.csv(csv_path, header=FALSE, col.names=c("x", "z_max"))
}
df$x <- suppressWarnings(as.numeric(df$x))
df$z_max <- suppressWarnings(as.numeric(df$z_max))
df <- df[is.finite(df$x) & is.finite(df$z_max), ]
df <- df[order(df$x), ]
if (nrow(df) < 3) stop("At least three finite profile points are required.")

xlim <- c(0, 60)
ylim <- c(0, 20)
xg <- seq(min(df$x), max(df$x), length = 500)

## ============================================================
## Modelle
## ============================================================
m_lm <- lm(z_max ~ x, data = df)
m0 <- lm(z_max ~ x, data = df)

seg1 <- tryCatch(segmented(m0, seg.Z = ~x, npsi = 1), error = function(e) NULL)
seg2 <- tryCatch(segmented(m0, seg.Z = ~x, npsi = 2), error = function(e) NULL)
seg3 <- tryCatch(segmented(m0, seg.Z = ~x, npsi = 3), error = function(e) NULL)
seg4 <- tryCatch(segmented(m0, seg.Z = ~x, npsi = 4), error = function(e) NULL)
seg5 <- tryCatch(segmented(m0, seg.Z = ~x, npsi = 5), error = function(e) NULL)

## ============================================================
## Ergebnisse extrahieren und als CSV speichern
## ============================================================
extract_results <- function(model, name, xg) {
  if (is.null(model)) return(NULL)
  
  if (name == "Linear") {
    slopes <- coef(model)["x"]
    kp <- numeric(0)
  } else {
    slopes <- slope(model)$x[, "Est."]
    kp <- model$psi[, "Est."]
  }
  intercept <- unname(coef(model)[1])
  
  angles <- atan(slopes) * 180/pi
  y_pred <- predict(model, newdata = data.frame(x = xg))
  
  data.frame(
    model = name,
    n_breakpoints = length(kp),
    breakpoints = paste(round(kp, 4), collapse = ";"),
    slopes = paste(round(slopes, 4), collapse = ";"),
    angles_deg = paste(round(angles, 2), collapse = ";"),
    n = nobs(model),
    k = length(coef(model)) + 1L,
    aic = AIC(model),
    bic = AIC(model, k = log(nobs(model))),
    aicc = if (nobs(model) > length(coef(model)) + 2L) {
      AIC(model) + 2 * (length(coef(model)) + 1L) *
        (length(coef(model)) + 2L) / (nobs(model) - length(coef(model)) - 2L)
    } else NA_real_,
    intercept = intercept,
    x_pred = paste(round(xg, 4), collapse = ";"),
    y_pred = paste(round(y_pred, 4), collapse = ";"),
    stringsAsFactors = FALSE
  )
}

results <- list(
  Linear = m_lm,
  "1 Segment" = seg1,
  "2 Segmente" = seg2,
  "3 Segmente" = seg3,
  "4 Segmente" = seg4,
  "5 Segmente" = seg5
)

# A failed segmented fit can return an ordinary lm without raising an error.
# Keep it out of the segmented exports; the Linear fit is already included.
for (name in setdiff(names(results), "Linear")) {
  if (!is.null(results[[name]]) && !inherits(results[[name]], "segmented")) {
    warning(paste("Skipping", name, "- segmented fit returned a non-segmented model"))
    results[name] <- list(NULL)
  }
}

# Legacy labels count breakpoints, not straight-line sections.
failed <- names(results)[vapply(results, is.null, logical(1))]
if (length(failed)) warning(paste("Failed models:", paste(failed, collapse=", ")))
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)
saveRDS(results, file.path(out_dir, "slope_fits.rds"))
write.csv(df, file.path(out_dir, "fit_upper_edge.csv"), row.names=FALSE)

out_df <- do.call(rbind, lapply(names(results), function(name) {
  extract_results(results[[name]], name, xg)
}))

out_path <- file.path(out_dir, "slope_results_R.csv")
write.csv(out_df, out_path, row.names=FALSE)
cat("R results saved to:", out_path, "\n")
