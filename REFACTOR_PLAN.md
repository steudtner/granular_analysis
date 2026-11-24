# Refactor status

## Supported workflows

- 2D porosity and upper-edge extraction
- z-slice profiles and exact ellipsoid slice plots
- monitor-based kinetic-energy analysis
- segmented-regression slope analysis through R
- collision-distance analysis
- 3D ellipsoid particle rendering (`renderview`)

## Architecture

- `paths.CasePaths` owns simulation-case path resolution.
- `scripts.run_all` is the supported master CLI and returns a failing exit code
  when a requested pipeline fails.
- Analysis caches include input and parameter fingerprints.
- Slice plotting and porosity analysis share periodic geometry, union-area, and
  upper-edge rules.
- PyVista is optional and imported only by the ellipsoid renderer.

## Removed workflows

The experimental 3D alpha-shape, open-top porosity, PCA-prism, and 3D/2D
mesh-comparison implementations were removed. They had conflicting algorithms,
broken imports, and no supported pipeline. This removal does not include the
3D ellipsoid particle renderer.

## Remaining compatibility code

`run_slice_analysis.py`, `run_depth_slice_analysis.py`, and
`analysis/slicing.py` remain for direct users, but are not part of the master
pipeline. New features should target the supported workflows above.
