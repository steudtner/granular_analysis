import shutil
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

use_latex = shutil.which("latex") is not None
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "text.usetex": use_latex,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.linewidth": 1.0,
})

from .plots import plot_shape, plot_profile, plot_kinetic_energy, plot_energy_time_series
