import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_shape(shape, out, title=""):
    fig, ax = plt.subplots()
    if shape is not None:
        ax.plot(shape[:, 0], shape[:, 1])
        ax.fill(shape[:, 0], shape[:, 1], alpha=0.2)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid()
    plt.tight_layout()
    plt.savefig(out, dpi=600)
    plt.close()


def plot_kinetic_energy(E, out, particle_ids=None, title="Kinetic energy per particle"):
    fig, ax = plt.subplots()
    x = np.arange(len(E)) if particle_ids is None else np.asarray(particle_ids)
    ax.plot(x, E, "-o")
    ax.set_xlabel("Particle")
    ax.set_ylabel("Kinetic energy")
    ax.set_title(title)
    ax.grid()
    plt.tight_layout()
    plt.savefig(out, dpi=600)
    plt.close()


def plot_energy_time_series(df, out, use_normalized=False):
    fig, ax = plt.subplots(figsize=(12, 4))

    if use_normalized and "time_norm" in df.columns and "kinetic_energy_norm" in df.columns:
        x = df["time_norm"]
        y = df["kinetic_energy_norm"]
        ax.set_xlabel(r"$t \sqrt{g}$", fontsize=14)
        ax.set_ylabel(r"$\sum E_{kin} / (\rho g D_{eq}^4)$", fontsize=14)
        ax.set_title("Normalized kinetic energy over time", fontsize=14)

        if len(y) > 0:
            E_max = y.max()
            threshold = 0.03 * E_max
            ax.axhline(
                y=threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label=r"$\epsilon = 3\% \sum E_{kin,max}$"
            )
    else:
        x = df["time"]
        y = df["kinetic_energy"]
        ax.set_xlabel("time", fontsize=12)
        ax.set_ylabel("kinetic energy", fontsize=12)
        ax.set_title("Total kinetic energy over time", fontsize=12)

    ax.plot(x, y, linewidth=2, label=r"$\sum E_{kin}$")
    ax.grid(True)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def plot_energy_histogram(particle_energies, timestep_label, out_path, normalization=1.0):
    E_norm = particle_energies / normalization

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(E_norm, bins=60)
    ax.set_yscale("log")
    ax.set_xlabel(r"$E^* = E / (\rho g D_{eq}^4)$", fontsize=12)
    ax.set_ylabel("count (log)", fontsize=12)
    ax.set_title(f"Normalized Energy Distribution\n{timestep_label}", fontsize=12)
    ax.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_profile(df, out):
    fig, ax = plt.subplots()
    ax.plot(df["z"], df["porosity"], "-o")
    ax.set_xlabel("z")
    ax.set_ylabel("porosity")
    ax.grid()
    plt.tight_layout()
    plt.savefig(out, dpi=600)
    plt.close()