#!/usr/bin/env bash
# Vom beliebigen Verzeichnis: vorhandene Daten aufbereiten und verfügbare Plots exportieren.
set -euo pipefail
THESIS_WORKSPACE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$THESIS_WORKSPACE"
export PYTHONPATH="$THESIS_WORKSPACE/granular_analysis/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/diplomarbeit-matplotlib"
export MPLBACKEND=Agg
THESIS_PYTHON="${THESIS_PYTHON:-$THESIS_WORKSPACE/miniforge3/primeframework/bin/python}"
"$THESIS_PYTHON" -m granular_analysis.thesis.aicc \
  --delta16 DiscretizationGranularCollapse/primeJainSU/results_granular_analysis/slope_analysis/slope_results_R.csv \
  --delta20 DiscretizationGranularCollapse/primeHigParMidFluid/results_granular_analysis/slope_analysis/slope_results_R.csv --legacy
"$THESIS_PYTHON" -m granular_analysis.thesis.prepare
"$THESIS_PYTHON" -m granular_analysis.thesis.plots kriterien bilder/generiert/kap7_aicc.csv
"$THESIS_PYTHON" -m granular_analysis.thesis.plots translation bilder/generiert/kap7_1_translation_eingabe.csv
"$THESIS_PYTHON" -m granular_analysis.thesis.plots alpha_flaeche bilder/generiert/kap6_2_alpha_flaeche_eingabe.csv
"$THESIS_PYTHON" -m granular_analysis.thesis.plots porositaet bilder/generiert/kap7_3_porositaet_eingabe.csv
"$THESIS_PYTHON" granular_analysis/thesis/report.py
