# src/granular_analysis/plotting/z_slice_plotter.py
"""
z_slice_plotter – Plot‑Utilities für 2‑D‑Schnitte aus Partikel‑Daten.

- Kein Diagrammtitel
- Legende nur, wenn eine Alpha‑Shape angezeigt wird
- X‑ und Y‑Achsenbeschriftung in LaTeX:  $x/D_p$  und  $y/D_p$
- Tick‑Abstand 5 (einheitlich für beide Achsen)
- Optionales Speichern des Plots (standardmäßig nach ``RESULTS_DIR`` wie im Rest des Packages)
"""

from __future__ import annotations

from granular_analysis.plotting import plotstyle

import pathlib
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Paketinterner Pfad, in dem alle Ergebnis‑Files abgelegt werden
# ----------------------------------------------------------------------
from granular_analysis.plotting.helpers import save_figure

# ----------------------------------------------------------------------
# Hilfsfunktion: Punkte, die zu einem gewünschten Z‑Schnitt gehören
# ----------------------------------------------------------------------
def _filter_by_z(particles: np.ndarray, z_val: float, eps: float = 1e-2) -> np.ndarray:
    """
    Gibt alle Partikel zurück, deren Z‑Koordinate innerhalb ``eps``
    um ``z_val`` liegt. Erwartet ein Nx3‑Array (x, y, z).
    """
    particles = np.asarray(particles)
    mask = np.abs(particles[:, 2] - z_val) <= eps
    return particles[mask][:, :2]          # nur x‑, y‑Koordinaten


# ----------------------------------------------------------------------
# Hauptfunktion: Plot eines Z‑Schnitts
# ----------------------------------------------------------------------
def plot_z_slice(
    particles: list | np.ndarray,
    z_value: float,
    *,
    with_alpha_shape: bool = False,
    alpha: float = 0.5,
    output_path: str | pathlib.Path | None = None,
) -> plt.Figure:
    """
    Plot einen 2‑D‑Schnitt bei ``z_value``.

    Parameters
    ----------
    particles
        Iterable von Partikeln (mindestens ``[x, y, z]``). Empfohlen ist ein
        ``np.ndarray`` mit Shape ``(N, 3)``.
    z_value
        Z‑Koordinate des gewünschten Schnitts.
    with_alpha_shape
        Wenn ``True`` wird eine Alpha‑Shape (``granular_analysis.geometry.alpha2d.compute_alpha_shape_2d``)
        aus den Schnitt‑Punkten berechnet und geplottet.
    alpha
        Alpha‑Parameter für die Alpha‑Shape (nur relevant, wenn ``with_alpha_shape`` ist).
    output_path
        Pfad, unter dem das Bild gespeichert werden soll.  Wenn ``None`` (Standard)
        wird automatisch ein Pfad in ``RESULTS_DIR`` gewählt, z. B.

        ``RESULTS_DIR / "slice_z2.5_alpha_true.png"``

    Returns
    -------
    matplotlib.figure.Figure
        Das erzeugte Figure‑Objekt (kann weiter bearbeitet werden).
    """
    # --------------------------------------------------------------
    # 1️⃣  Daten filtern / vorbereiten
    # --------------------------------------------------------------
    pts = np.asarray(particles)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(
            "particles must be an array of shape (N,2) or (N,3)"
        )

    if pts.shape[1] == 3:
        pts_xy = _filter_by_z(pts, z_value)
    else:
        pts_xy = pts

    if pts_xy.size == 0:
        raise ValueError(
            "Keine Partikel gefunden für den gegebenen Schnitt bzw. die übergebenen Punkte"
        )

    # --------------------------------------------------------------
    # 2️⃣  Matplotlib‑Figure anlegen
    # --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(plotstyle.figwidth_full, 6))

    # --------------------------------------------------------------
    # 3️⃣  Plot‑Logik
    # --------------------------------------------------------------
    if with_alpha_shape:
        # Alpha‑Shape nur bei Bedarf importieren
        from granular_analysis.geometry.alpha2d import compute_alpha_shape_2d

        shape_coords, _ = compute_alpha_shape_2d(pts_xy, alpha=alpha)

        if shape_coords is not None and shape_coords.size > 0:
            # Polygon (Alpha‑Shape) einzeichnen & füllen
            ax.plot(
                shape_coords[:, 0],
                shape_coords[:, 1],
                color="tab:blue",
                linewidth=1.2,
                label="Alpha Shape",
            )
            ax.fill(
                shape_coords[:, 0],
                shape_coords[:, 1],
                color="tab:blue",
                alpha=0.2,
            )
            # Legende nur bei Alpha‑Shape
            plotstyle.legend_above(ax)
        else:
            # Fallback: nur Punkte
            ax.scatter(pts_xy[:, 0], pts_xy[:, 1], c="black", s=20, edgecolor="none")
    else:
        # ------------------ ohne Alpha‑Shape --------------------
        ax.scatter(pts_xy[:, 0], pts_xy[:, 1], c="black", s=20, edgecolor="none")

        # Achsenbeschriftungen in LaTeX
        ax.set_xlabel(r"$x/D_p$", fontsize=10)
        ax.set_ylabel(r"$y/D_p$", fontsize=10)

        # Ticks alle 5 Einheiten (automatisch an Daten anpassen)
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        x_ticks = np.arange(
            np.floor(x_min / 5) * 5,
            np.ceil(x_max / 5) * 5 + 1,
            5,
        )
        y_ticks = np.arange(
            np.floor(y_min / 5) * 5,
            np.ceil(y_max / 5) * 5 + 1,
            5,
        )
        ax.set_xticks(x_ticks[x_ticks >= x_min])
        ax.set_yticks(y_ticks[y_ticks >= y_min])

    # --------------------------------------------------------------
    # 4️⃣  Gemeinsame Stil‑Elemente
    # --------------------------------------------------------------
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_title("")                     # keine Überschrift
    plotstyle.prepare_figure(plt.gcf())
    plt.tight_layout()

    # --------------------------------------------------------------
    # 5️⃣  Ausgabe bzw. Anzeige
    # --------------------------------------------------------------
    if output_path is None:
        default_name = f"slice_z{z_value}_alpha_{str(with_alpha_shape).lower()}.png"
        saved = save_figure(fig, None, default_name=default_name, dpi=600)
    else:
        saved = save_figure(fig, pathlib.Path(output_path), dpi=600)

    print(f"Plot gespeichert unter: {saved}")
    return fig


# ----------------------------------------------------------------------
# CLI‑Eintrag (ermöglicht Aufruf via ``python -m granular_analysis.plotting.z_slice_plotter``)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot a xy‑slice of particles at a given z‑value."
    )
    parser.add_argument(
        "--z",
        type=float,
        required=True,
        help="Z‑Koordinate des Schnitts",
    )
    parser.add_argument(
        "--timestep",
        type=str,
        default="latest",
        help="Zeit‑Schritt‑Marker (nur für Namensgebung)",
    )
    parser.add_argument(
        "--alphashape",
        type=str,
        default="FALSE",
        help="TRUE → Alpha‑Shape plotten, FALSE → nur Punkte",
    )
    args = parser.parse_args()

    # --------------------------------------------------------------
    # Demo‑Daten – ersetze diesen Block durch deine echte Daten‑Lader‑Funktion
    # --------------------------------------------------------------
    demo_particles = np.array(
        [
            [0.5, 1.2, 2.4],
            [1.8, 2.9, 2.5],
            [2.3, 0.7, 2.6],
            [3.1, 3.4, 2.5],
            [4.0, 1.0, 2.5],
        ]
    )

    # --------------------------------------------------------------
    # Plot‑Aufruf
    # --------------------------------------------------------------
    plot_z_slice(
        demo_particles,
        z_value=args.z,
        with_alpha_shape=args.alphashape.upper() == "TRUE",
        output_path=None,                     # nutzt das automatische RESULTS_DIR‑Verzeichnis
    )