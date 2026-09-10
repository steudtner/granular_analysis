#!/bin/bash
# ============================================================
# Automatisches Start-Skript für die Slope-Analyse (Python + R)
# ============================================================

# 1. Module laden (wichtig: in dieser Reihenfolge)
module purge
module load release/2026
module load GCC/14.2.0
module load R/4.5.1

# 2. Python Virtualenv aktivieren
source /data/horse/ws/s4713238-DA/envRomeo/bin/activate

# 3. In das Verzeichnis des Simulationsfalls wechseln
cd /data/horse/ws/s4713238-DA/DiscretizationGranularCollapse/primeJainSU

# 4. Pipeline starten (nutzt explizit den Python der Virtualenv,
#    um Konflikte mit dem System-Python zu vermeiden)
echo "Starting granular analysis pipeline..."

# Optional first argument: z plane. If not provided, use LZ/2 from the package config.
if [ -n "$1" ]; then
	Z_VAL="$1"
else
	Z_VAL=$( /data/horse/ws/s4713238-DA/envRomeo/bin/python -c "from granular_analysis.config import LZ; print(float(LZ)/2.0)" )
fi

# Configure densify and upper-edge bins (change values here if desired)
UPPER_EDGE_BINS=80
DENSIFY=0.5

/data/horse/ws/s4713238-DA/envRomeo/bin/python -m granular_analysis.scripts.run_all \
	--only 2d slope --force --alphashape --upper_edge \
	--upper_edge_bins ${UPPER_EDGE_BINS} --densify ${DENSIFY} --z ${Z_VAL}

echo "Done."
