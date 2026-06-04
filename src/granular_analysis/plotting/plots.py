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


def plot_energy_time_series(df, out, x_col="time", y_col="kinetic_energy", title="Total kinetic energy over time"):

    fig, ax = plt.subplots()

    ax.plot(df[x_col], df[y_col], "-o")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid()

    plt.tight_layout()
    plt.savefig(out, dpi=600)
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