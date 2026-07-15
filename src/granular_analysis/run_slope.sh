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
/data/horse/ws/s4713238-DA/envRomeo/bin/python -m granular_analysis.scripts.run_all --only 2d slope --force

echo "Done."
