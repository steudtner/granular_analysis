#!/usr/bin/env python3
"""
run_all – führt sämtliche Analysen aus und erstellt für jede
Z‑Slice ein Plot (inkl. Alpha‑Shape) über das paketinterne
z_slice_plotter.  Alle Ausgaben landen im einheitlichen
RESULTS_DIR (z. B.  results_granular_analysis/).
"""

import argparse
import os
import pathlib

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Paketinterne Imports
# ----------------------------------------------------------------------
from granular_analysis.config import RESULTS_DIR                # ← Ergebnis‑Ordner
from granular_analysis.io.loader import read_ellipsoid_trn   # ← dein Daten‑Lader
from granular_analysis.plotting.slice_plotter import plot_exact_slice_from_ellipsoids

# ----------------------------------------------------------------------
# Konstanten (können über CLI‑Argumente überschrieben werden)
# ----------------------------------------------------------------------
DEFAULTS = {
    "folder": "/data/horse/ws/s4713238-DA/DiscretizationGranularCollapse/primeJainSU/results/ellipsoid/ellipsoid_trn",
    "alpha": 0.6,
    "min_spread": 1e-8,
    "skip_points": 1000,
    "z": 2.5,
    "xlim": (0, 60),
    "ylim": (0, 20),
}


# ----------------------------------------------------------------------
# Hilfsfunktion: CSV‑Export der Slice‑Statistik
# ----------------------------------------------------------------------
def _save_results(results: list, out_dir: pathlib.Path) -> pathlib.Path:
    df = pd.DataFrame(results)
    csv_path = out_dir / "alphashape_slice_areas.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved → {csv_path}")
    return csv_path


# ----------------------------------------------------------------------
# Haupt‑Workflow
# ----------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    if getattr(args, "only", None):
        print(f"Compatibility mode: --only {args.only}")
    if getattr(args, "force", False):
        print("Compatibility mode: --force enabled")

    # --------------------------------------------------------------
    # 1️⃣  Parameter aus CLI / Defaults
    # --------------------------------------------------------------
    folder      = args.folder or DEFAULTS["folder"]
    alpha       = args.alpha
    min_spread  = args.min_spread
    skip_points = args.skip_points
    z_plane     = args.z if args.z is not None else DEFAULTS["z"]
    xlim        = tuple(args.xlim) if args.xlim else DEFAULTS["xlim"]
    ylim        = tuple(args.ylim) if args.ylim else DEFAULTS["ylim"]

    # --------------------------------------------------------------
    # 2️⃣  Ergebnis‑Verzeichnis anlegen (einmalig)
    # --------------------------------------------------------------
    out_dir = pathlib.Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Alle Plots & CSV werden nach {out_dir} geschrieben.\n")

    # --------------------------------------------------------------
    # 3️⃣  Letzte *.bin‑Datei im angegebenen Ordner bestimmen
    # --------------------------------------------------------------
    bin_files = sorted(f for f in os.listdir(folder) if f.endswith(".bin"))
    if not bin_files:
        raise RuntimeError("Keine .bin‑Dateien im Ordner gefunden.")
    last_file = bin_files[-1]
    bin_path = os.path.join(folder, last_file)
    print(f"Nutze aktuellsten Zeitschritt: {last_file}")

    # --------------------------------------------------------------
    # 4️⃣  Daten laden (echte Partikel‑Koordinaten)
    # --------------------------------------------------------------
    ellipsoids = read_ellipsoid_trn(bin_path)

    # Wir benötigen nur die Zentrum‑Koordinaten (x, y, z)
    pts_all = np.array([e["x"] for e in ellipsoids])
    pts_all = pts_all[np.isfinite(pts_all).all(axis=1)]

    if pts_all.shape[0] < 5:
        raise RuntimeError("Zu wenige Punkte für die Analyse vorhanden.")

    # Als Alternative für die Alphashape: überspringe nur die ersten Ellipsoide
    ellipsoids_alpha = ellipsoids[skip_points:] if len(ellipsoids) > skip_points else []

    # Ergebnis‑Liste für CSV
    results = []

    # --------------------------------------------------------------
    # 5️⃣  Einzel‑Slice‑Plot auf exakter z‑Ebene
    # --------------------------------------------------------------
    print(f"\n--- Exact slice at z = {z_plane:.3f} ---")

    plot_result = plot_exact_slice_from_ellipsoids(
        ellipsoids,
        z_plane,
        output_path=out_dir / f"alphashape_slice_{z_plane:.2f}.png",
        alphashape=args.alphashape,
        alpha=alpha,
        upper_edge=args.upper_edge,
        upper_edge_bins=args.upper_edge_bins,
        skip_points=skip_points,
        xlim=xlim,
        ylim=ylim,
    )
    print(f"Plot gespeichert unter: {plot_result['output_path']}")

    results.append({
        "z": z_plane,
        "area": plot_result["area"],
        "num_particles": plot_result["num_ellipses"],
        "alpha_used_particles": plot_result["num_alpha_points"],
    })

    # --------------------------------------------------------------
    # 7️⃣  CSV‑Datei erzeugen
    # --------------------------------------------------------------
    _save_results(results, out_dir)

    print("\n=== Alle Analysen abgeschlossen ===")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full analysis pipeline and plot Z‑slices "
                    "(mit optionaler Alpha‑Shape)."
    )
    # Pfad‑Parameter (kann überschrieben werden)
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Verzeichnis, in dem *.bin‑Dateien liegen (Standard: aus DEFAULTS).",
    )
    # Alpha‑Shape‑Parameter
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULTS["alpha"],
        help="Alpha‑Parameter für Alphashape (Standard 0.6).",
    )
    parser.add_argument(
        "--alphashape",
        action="store_true",
        help="Alpha‑Shape beim Slice‑Plot aktivieren (Standard = OFF).",
    )
    parser.add_argument(
        "--upper_edge",
        action="store_true",
        help="Obere Kante der Alpha‑Shape im Plot markieren.",
    )
    parser.add_argument(
        "--upper_edge_bins",
        type=int,
        default=50,
        help="Anzahl Bins für die Oberkanten‑Extraktion.",
    )
    # Numerische Stabilisierung & Daten‑Reduktion
    parser.add_argument(
        "--min_spread",
        type=float,
        default=DEFAULTS["min_spread"],
        help="Kleinste zufällige Verschiebung, falls Punkte kollinear sind.",
    )
    parser.add_argument(
        "--skip_points",
        type=int,
        default=DEFAULTS["skip_points"],
        help="Anzahl initialer Punkte, die übersprungen werden (wie im Original).",
    )
    # Exakter Z‑Schnitt
    parser.add_argument(
        "--z",
        type=float,
        default=DEFAULTS["z"],
        help="Z‑Ebene für den exakten Schnitt.",
    )
    # Plot‑Limits (optional über CLI überschreibbar)
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=DEFAULTS["xlim"],
        help="X‑Achsen‑Grenzen für alle Plots (z. B. --xlim 0 60).",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=DEFAULTS["ylim"],
        help="Y‑Achsen‑Grenzen für alle Plots (z. B. --ylim 0 20).",
    )

    # Backwards-compatible flags used by older wrapper scripts
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        choices=["2d", "slope", "all", "slice"],
        help="Compatibility flag for older wrappers; accepted but currently not required.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Compatibility flag for older wrappers.",
    )

    args = parser.parse_args()
    main(args)