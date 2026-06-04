from granular_analysis.scripts.run_2d_analysis import main as run2d
from granular_analysis.scripts.run_3d_porosity import main as run3d
from granular_analysis.scripts.run_slice_comparison import main as run_compare

from granular_analysis.utils.logging_utils import suppress_noise
from granular_analysis.config import RESULTS_DIR


def main():
    suppress_noise()

    print("=" * 32)
    print("Running 2D analysis...")
    print("=" * 32)
    run2d()

    print("\n" + "=" * 32)
    print("Running 3D analysis...")
    print("=" * 32)
    run3d()

    print("\n" + "=" * 32)
    print("Running slice comparison...")
    print("=" * 32)
    run_compare()

    print("\n" + "=" * 32)
    print(f"DONE — outputs in: {RESULTS_DIR}")
    print("=" * 32)


if __name__ == "__main__":
    main()