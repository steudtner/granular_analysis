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

# CSV ohne Header einlesen (Basis-R)
df <- read.csv(csv_path, header=FALSE, col.names=c("x", "z_max"))
df <- df[is.finite(df$x) & is.finite(df$z_max), ]
df <- df[order(df$x), ]

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
  
  angles <- atan(slopes) * 180/pi
  y_pred <- predict(model, newdata = data.frame(x = xg))
  
  data.frame(
    model = name,
    n_breakpoints = length(kp),
    breakpoints = paste(round(kp, 4), collapse = ";"),
    slopes = paste(round(slopes, 4), collapse = ";"),
    angles_deg = paste(round(angles, 2), collapse = ";"),
    aic = AIC(model),
    bic = BIC(model),
    x_pred = paste(round(xg, 4), collapse = ";"),
    y_pred = paste(round(y_pred, 4), collapse = ";"),
    stringsAsFactors = FALSE
  )
}

results <- list(
  Linear = m_lm,
  "1 Segment" = seg1,
  "2 Segmente" = seg2,
  "3 Segmente" = seg3
)

out_df <- do.call(rbind, lapply(names(results), function(name) {
  extract_results(results[[name]], name, xg)
}))

out_path <- file.path(out_dir, "slope_results_R.csv")
write.csv(out_df, out_path, row.names=FALSE)
cat("R results saved to:", out_path, "\n")
