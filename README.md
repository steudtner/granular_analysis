# granular-analysis

Post-processing tools for ellipsoid granular simulations.

`granular-analyze --only 2d` exports the selected frame's Böschungskante
to `2d/alphashape_last_timestep.csv` and surface dimensions to
`2d/surface_metrics.csv`, and prints the dimensions to the console.
The CSV contains an `initial` row for the earliest available TRN frame and a
`selected` row for the analyzed frame, using the same slice and alpha parameters.
The `frame` column identifies their source files; the earliest available frame
may be later than simulation time zero if the initial output is missing.
`max_runout_length` is the alpha-shape's full horizontal extent (`x_max - x_min`);
`remaining_height` is its highest y-coordinate above the reference plane `y=0`.
Both use simulation length units and the selected z-slice. They are dimensions
of each reported frame (selected: latest by default), not maxima over time or across z.
The metrics CSV includes the source frame and slice position.

Monitor histories stored in `results/ellipsoid/history/ell_*_monitor.dat`
are read before the corresponding live monitor files. At overlapping times,
the live data take precedence. Optional `results/history/rst_info.dat` records
are also included when matching fluid checkpoints. This permits complete
analysis histories without rewriting files that a running simulation appends.

## 2D Fluid Overlays

```bash
granular-analyze --only 2d --fluid
granular-analyze --only 2d --fluid u --counter 3961 --z 2.5
granular-analyze --only 2d --fluid pressure --counter 3961 --z 2.5
granular-analyze --only 2d --fluid v --time 91.0 --z 2.25
```

`--fluid` defaults to velocity magnitude; `--fluid u`, `v`, or `w` selects
a signed velocity component. `--fluid pressure` reads the cell-centered
pressure from `pre_*.bin` and displays `p/(rho_w*g*D_EQ)`, using configured
`D_EQ` as the reference diameter and `rho_f` as the
water/fluid density and the magnitude of `(grx, gry, grz)` from the case's
`input.dat`. Density and gravity must be positive. No staggered velocity
averaging is applied to pressure. Particle interiors are opaque and do not show
the fluid. The staggered velocity components are averaged to cell centers
and spatially interpolated to the requested z-plane.

Only equal stored simulation times are combined. Without `--counter`, the
latest particle frame with an exact fluid checkpoint match is selected.
With `--counter`, a missing exact match is an error; no nearby fluid timestep
is substituted. This is checked before analysis, with a suggested matching
counter. `--time T` instead selects the nearest available common time and
reports the actual time and both counters; fluid and particles still match
exactly. Without `--fluid`, it selects the nearest particle frame. `--time`
and `--counter` are mutually exclusive. `--counter` always refers to the particle TRN file, not the
fluid checkpoint number. Native PRIME real64 PETSc velocity files and the
matching `rst_info.dat` and grid files are required.

2D slice figures include the particle counter, z-plane, and selected fluid
component, for example `alphashape_slice_003961_z2.50_u.svg`.

## Install

Run these commands from the repository directory:

```bash
python -m pip install -e .
python -m pip install -e ".[render]"    # optional PyVista particle renderer
python -m pip install -e ".[mesh]"      # optional mesh utilities
python -m pip install -e ".[progress]"  # optional progress bars
python -m pip install -e ".[dev]"       # optional test/development tools
```

## Quick start and automatic path detection

`--path` is optional. When you start the command in a simulation case
directory, the case is detected from the current directory:

```bash
cd /path/to/case
granular-analyze --only 2d slices
```

To select a different case explicitly:

```bash
granular-analyze --path /path/to/case --only 2d slices
```

Input is resolved in this order:

1. `--path PATH`
2. `GRANULAR_DATA_FOLDER`
3. the current working directory

A path may identify the case root, `results`, `ellipsoid`,
`ellipsoid_trn`, `ellipsoid_coll`, or `ellipsoid_monitor`:

```bash
granular-analyze --path /path/to/case --only 2d
granular-analyze --path /path/to/case/results --only 2d
granular-analyze --path /path/to/case/results/ellipsoid --only 2d
granular-analyze --path /path/to/case/results/ellipsoid/ellipsoid_trn --only 2d
```

Environment variables can make paths reusable:

```bash
export GRANULAR_DATA_FOLDER=/path/to/case
export GRANULAR_RESULTS_DIR=/path/to/analysis-output
granular-analyze --only 2d slices
```

Output defaults to `./results_granular_analysis`, relative to the directory
where the command starts. `--out` overrides `GRANULAR_RESULTS_DIR`.

## Pipelines

| Name | What it does |
| --- | --- |
| `2d` | Computes porosity and the upper edge at one z-plane from the latest TRN frame. |
| `slices` | Computes area and porosity profiles across configured z-levels. |
| `porosity_time` | Opt-in time-coloured z profiles and mean porosity ± spatial standard deviation over time. |
| `energy` | Computes the energy history from transient and monitor data. |
| `slope` | Runs the packaged R slope analysis on the 2D output. Requires `Rscript`, or the executable selected by `RSCRIPT`. |
| `collision_distances` | Computes collision/contact distances for a frame with TRN data and all three collision RK stages. |
| `renderview` | Produces isometric, top, front, and side PNGs plus matching PDF/SVG containers. Requires the `render` dependency group. |
| `particle_fluid` | Combines frontal particle rendering and a synchronous 2D fluid slice, with fixed x/y limits. Requires `render`. |
| `alpha_sensitivity` | Sweeps alpha-shape area and porosity over an alpha range, then writes a curve plot, a panel of alpha shapes, and a CSV. |
| `slope_alpha_sensitivity` | Runs the 2D upper-edge and R slope-model pipeline for every alpha value, then compares model AIC/BIC across the range. |

```bash
granular-analyze                              # run all pipelines
granular-analyze --only 2d slices energy     # run a selected set
granular-analyze --exclude slope renderview  # run all except a selected set
```

The two sensitivity pipelines are opt-in and are not included by the bare
`granular-analyze` command, because they run multiple analyses.

## All command-line options

General form:

```bash
granular-analyze [OPTIONS]
```

### Selection, paths, and execution

| Option | Meaning | Copy-ready example |
| --- | --- | --- |
| `-h`, `--help` | Show built-in help and exit. | `granular-analyze --help` |
| `--only NAME [NAME ...]` | Run only these pipelines. Valid names: `2d`, `slices`, `energy`, `slope`, `collision_distances`, `renderview`, `alpha_sensitivity`, `slope_alpha_sensitivity`. | `granular-analyze --only 2d slices` |
| `--exclude NAME [NAME ...]` | Remove these pipelines from the selected set. | `granular-analyze --exclude slope renderview` |
| `--path PATH` | Select a case root or supported results subdirectory. Optional when inferred from the environment or current directory. | `granular-analyze --path /data/run42 --only 2d` |
| `--folder PATH` | Deprecated alias for `--path`. Never supply both. | `granular-analyze --folder /data/run42 --only 2d` |
| `--out DIR` | Write analysis files under `DIR`. | `granular-analyze --out /tmp/run42-analysis --only 2d slices` |
| `--counter INTEGER` | Select a frame for `slices`, `collision_distances`, and `renderview`, or a zero-based monitor timestep index for an `energy` histogram. | `granular-analyze --only energy --counter 1682` |
| `--histogram-timesteps INDICES` | Comma-separated monitor timestep indices for energy histograms; overrides `--counter` for energy. Negative indices count from the end. | `granular-analyze --only energy --histogram-timesteps 0,100,-1` |
| `--force` | Recompute even when a valid cache or render exists. | `granular-analyze --only 2d slices --force` |
| `--continue-on-error` | Continue after a failed pipeline. The final exit status remains unsuccessful if any step fails. | `granular-analyze --continue-on-error` |
| `--dry` | Show the resolved case, directories, pipelines, and counter without writing files. | `granular-analyze --only 2d slices --dry` |

### Slice and plot controls

| Option | Meaning | Copy-ready example |
| --- | --- | --- |
| `--z FLOAT` | Set the plane for `2d`, the exact-slice plot, and sensitivity pipelines. The default is `Lz / 2`. | `granular-analyze --only 2d slices --z 5.0` |
| `--lz FLOAT` | Set the periodic z-length used by `2d` and both sensitivity pipelines. Defaults to the configured `LZ`. | `granular-analyze --only alpha_sensitivity --lz 2.5` |
| `--alpha FLOAT` | Set the alpha-shape parameter for a single `2d`, `slices`, or exact-slice analysis. | `granular-analyze --only 2d --alpha 0.35` |
| `--alpha-start FLOAT` | First alpha value for a sensitivity sweep. Default: `0.1`. | `granular-analyze --only alpha_sensitivity --alpha-start 0.2` |
| `--alpha-stop FLOAT` | Last alpha value for a sensitivity sweep. Default: `1.0`. | `granular-analyze --only alpha_sensitivity --alpha-stop 2.0` |
| `--alpha-step FLOAT` | Alpha increment for a sensitivity sweep. Default: `0.1`. | `granular-analyze --only alpha_sensitivity --alpha-step 0.05` |
| `--alphashape` | Draw the alpha-shape in the exact-slice plot. With `--only 2d`, also request that plot. | `granular-analyze --only 2d --alphashape` |
| `--upper-edge`, `--upper_edge` | Draw the binned upper edge and alpha-shape in the exact-slice plot. With `--only 2d`, also request that plot. | `granular-analyze --only 2d --upper-edge` |
| `--upper-edge-bins INTEGER`, `--upper_edge_bins INTEGER` | Set upper-edge x-bin count. Default: `50`. | `granular-analyze --only 2d slices --upper-edge-bins 80` |
| `--densify FLOAT` | Set the maximum boundary-segment length when densifying slice geometry. | `granular-analyze --only 2d slices --densify 0.05` |
| `--skip-points INTEGER`, `--skip_points INTEGER` | Ignore the first N particles in `slices` and the exact-slice plot. | `granular-analyze --only slices --skip-points 4` |
| `--xlim XMIN XMAX` | Set exact-slice plot x-axis limits. | `granular-analyze --only slices --xlim 0 60` |
| `--ylim YMIN YMAX` | Set exact-slice plot y-axis limits. | `granular-analyze --only slices --ylim 0 20` |

The underscore spellings are compatibility aliases. Prefer hyphenated forms
in new commands.

## Copy-ready recipes

Default 2D result using automatic path detection:

```bash
cd /path/to/case
granular-analyze --only 2d
```

Detailed slice profile and exact-slice plot:

```bash
cd /path/to/case
granular-analyze \
  --only slices \
  --counter 1200 \
  --alpha 0.35 \
  --densify 0.05 \
  --skip-points 4 \
  --upper-edge \
  --upper-edge-bins 80 \
  --xlim 0 60 \
  --ylim 0 20
```

One z-plane with a custom output directory:

```bash
granular-analyze \
  --path /path/to/case \
  --out /path/to/analysis-output \
  --only 2d slices \
  --z 5.0 \
  --alpha 0.35 \
  --upper-edge
```

Energy only:

```bash
granular-analyze --path /path/to/case --only energy
```

Energy histogram for a specific timestep (zero-based monitor row index):

```bash
granular-analyze --only energy --counter 1682
granular-analyze --only energy --histogram-timesteps 0,100,1682,-1
```

Energy analysis excludes a particle's rotational contribution at samples where
it exceeds 50% of the unfiltered rotational energy of the selected particles.
Set `--rotation-share-threshold 0.8` for an 80% threshold, or
`--rotation-share-threshold 1` to disable the filter. The supported range is
0.5 to 1; the comparison is strictly greater than the threshold and is applied
once against the raw total. Translation is retained. Total energy, ratios and
histograms use the filtered rotation. This is an analysis exclusion, not a
correction of the simulated particle state.
The CSV retains `rotational_energy_raw`, `rotational_energy_raw_norm`,
`rotational_energy_excluded`, `rotation_excluded_monitor` (zero means none),
and `rotation_share_threshold`. Changing the threshold invalidates the cache.

The energy history is still computed in full. Without either selection option,
the histogram uses the final timestep. Histograms are saved under
`results_granular_analysis/energy_histograms/`.

2D followed by slope analysis:

```bash
cd /path/to/case
granular-analyze --only 2d slope
```

Custom R executable:

```bash
RSCRIPT=/path/to/Rscript granular-analyze --only 2d slope
```

Alpha-shape area and porosity sensitivity at a single plane:

```bash
granular-analyze \
  --path /path/to/case \
  --out /path/to/analysis-output \
  --only alpha_sensitivity \
  --z 1.25 \
  --lz 2.5 \
  --alpha-start 0.1 \
  --alpha-stop 2.0 \
  --alpha-step 0.1
```

Full upper-edge and slope-model sensitivity analysis:

```bash
granular-analyze \
  --path /path/to/case \
  --out /path/to/analysis-output \
  --only slope_alpha_sensitivity \
  --z 1.25 \
  --lz 2.5 \
  --alpha-start 0.1 \
  --alpha-stop 2.0 \
  --alpha-step 0.1
```

The first command writes `alpha_sensitivity/alpha_sensitivity.{png,svg,csv}`
and `alpha_sensitivity/alpha_shapes.{png,svg}`. The second writes one
`slope_alpha_sensitivity/alpha_<value>/` directory per alpha, each containing
the upper-edge data, R fit results, and annotated model plots. Its combined
comparison is in `slope_alpha_sensitivity/model_selection/` as PNG, SVG, and
CSV files. In both sweeps, lower AIC and BIC values indicate the preferred
model; BIC penalizes additional breakpoints more strongly.

Compare R's `segmented` models with the experimental Python/SciPy continuous
piecewise-linear models on exactly the same upper-edge data:

```bash
cd /path/to/case
granular-analyze --only 2d
granular-compare-slopes
```

On a cluster, load R first or pass the executable explicitly:

```bash
module load R
granular-compare-slopes --rscript "$(command -v Rscript)"
```

Use an existing CSV or change the maximum model complexity:

```bash
granular-compare-slopes \
  --csv /path/to/alphashape_last_timestep.csv \
  --out /path/to/slope-comparison \
  --max-breakpoints 3
```

The comparison produces `slope_results_R.csv`,
`slope_results_python.csv`, `slope_method_comparison.csv`, and separate plot
directories for both methods. The normal `slope` pipeline continues to use R;
Python is an explicit comparison method.

Collision distances at a known complete frame:

```bash
granular-analyze \
  --path /path/to/case \
  --only collision_distances \
  --counter 1200
```

Four rendered particle views:

```bash
python -m pip install -e ".[render]"
granular-analyze \
  --path /path/to/case \
  --only renderview \
  --counter 1200
```

Force recomputation and continue through independent failures:

```bash
granular-analyze \
  --only 2d slices energy \
  --force \
  --continue-on-error
```

Preview a full run:

```bash
granular-analyze --path /path/to/case --dry
```

The authoritative short option list is always available with:

```bash
granular-analyze --help
```

## Energy comparison

Overlay the current/primary case with one or more other cases:

```bash
granular-analyze --only energy --path prolaten/primeCollNick \
  --compare prolaten/new-prime-gran-coll-2000Uncut_fp_collision
```

Paths are relative to your working directory (or use absolute paths). The
primary case is `--path`, or the current case if omitted. Optional
`--compare-labels Nick Uncut` sets legend labels in case order.
Use `--energy-component translational` or `rotational` to select a component;
the default `kinetic` includes both. Default axes use the existing per-case
energy and time normalization; `--energy-raw` plots unnormalized quantities.
Each case retains its native time samples and full duration: no interpolation,
time shifting, clipping to shared duration, or extrapolation is performed.

Each case's energy routine validates its cache or recomputes from monitors.
`--force` forces recomputation. Comparison PNG/SVG, combined CSV and a JSON
case manifest are written under `results_granular_analysis/comparison/energy`
(or `--out` followed by `comparison/energy`). Frame/time selectors are not
applicable to energy-only comparisons; in mixed comparisons they select only
the geometry frames, while energy retains the full curve.

With `--potential-energy`, energy time-series, component and comparison plots
include a dashed reference (hidden by default) for total
initial potential energy, `sum(m_i * |g| * y_i)`, using retained initial input
particle y coordinates (column 7), densities and semiaxes. Zero potential is at
`y=0`; no buoyancy correction or final-bed energy subtraction
is applied. This is an input-configuration reference, not the restart state's
potential energy, and may represent more particles than available monitors.
Energy analysis always applies `SKIP_FIRST_PARTICLES` by default, independently
of the geometry skip toggle. `--skip-points N` overrides the count, including
in comparisons. The first N input rows (not physical IDs) are excluded from
potential energy, monitored kinetic energy, and histograms. The selected count
is recorded in the energy CSV and cache fingerprint.
Comparison plots show a reference in each case's curve color. The reference
uses the same normalization as the curve and is included in the energy CSV.
Missing initial positions cause an explicit warning and omission of the line.

## Geometry comparison

```bash
granular-analyze --only 2d slices slope --path prolaten/primeCollNick \
  --compare prolaten/new-prime-gran-coll-2000Uncut_fp_collision \
  --time 90 --z 2.25 --compare-labels Nick Uncut
```

`2d` overlays alpha contours and upper edges and compares slice porosity.
`slices` overlays porosity and occupied/alpha areas over the configured z levels.
`slope` creates fresh upper edges at the selected frame, invokes the existing R
segmented regression, and overlays fits and segment angles for each breakpoint
count successful in all cases. It also exports all model coefficients and
AIC/BIC; no different model orders are silently treated as equivalent.
R and the usual slope packages must be available for `slope`.

`--time` chooses the nearest frame independently in each case; `--counter`
selects the same counter number, which need not denote the same physical time.
Without either option each case uses its latest frame. Actual times/counters
appear in legends and `comparison.json`; differing times trigger a notice.
States are compared, never merged or temporally interpolated.
Common `--alpha`, `--lz`, `--skip-points`, `--densify`, and
`--upper-edge-bins` settings are applied to all cases. Outputs are under
`comparison/geometry/<selection>_z<z>/`, including PNG/SVG overlays and CSVs.
Per-case intermediate slope files are isolated inside that directory.
Add `energy` to `--only` for a combined run. Other pipelines and fluid overlays
are not currently supported by `--compare`.

## Fluid vortex criteria

Exact 2D particle slices colour particle interiors by their full 3D speed
`sqrt(u² + v² + w²)` by default. The linear colour scale runs from white at
zero to red at the highest finite speed in the slice, with a labelled colour
bar. The scale is frame-specific. Missing or invalid velocities appear white.
Use `--no-velocity-colors` to disable colouring (`--no-velocity-arrows` remains
an alias). Alpha-contour comparison plots do not show individual particles.

Use `granular-analyze --only 2d --fluid lambda2 --time 90 --z 2.25`
or `--fluid q` (also accepts `Q`). Both use the full 3D velocity gradient,
including derivatives normal to the displayed slice, at the matched checkpoint.
The staggered velocities are centered as for the existing velocity overlay.
Derivatives use cell-center coordinates, periodic neighbors where configured,
and one-sided differences at nonperiodic edges. At least two cells per axis
are required; three or more allow second-order edge differences.
Gradients are linearly interpolated to the requested z before evaluating the
criterion. Only nearby z planes are read, not the full velocity volume.

For `A = grad(u)`, `S = (A+A.T)/2`, `Omega = (A-A.T)/2`:
`Q = (||Omega||^2 - ||S||^2)/2`; lambda2 is the middle eigenvalue of
`S@S + Omega@Omega`. Vortex regions have Q > 0 or lambda2 < 0, respectively.
Values have inverse simulation-time squared units and are not pressure-normalized.
Both vortex fields use a symmetric logarithmic color scale (base 10), preserving
negative values and zero. The linear region extends to 0.1% of the slice's
maximum absolute value on either side of zero; color limits are symmetric.
Limits are selected per slice, so colors need not represent equal values across
different frames. Velocity and pressure color scales are unchanged.
Particle interiors remain masked. Near solid interfaces, interpret velocity
derivatives cautiously; masking does not remove solid values from the stencil.

## Velocity-axis alignment

Analyze instantaneous TRN translational velocity against the longest semiaxis:

```bash
granular-analyze --only alignment --counter 660
granular-analyze --only alignment --alignment-mode oblate --time 90
```

Without a counter or time, the latest particle frame is used. `--time` selects
the nearest stored particle frame. The quaternion rotates the local principal
axis into the global velocity coordinate system. Axes are undirected:
prolate angles are `acos(abs(v_hat dot longest_axis))` in degrees (0 to 90).
Oblate mode uses the stored c axis directly (range 0 to 90 degrees):
0 means motion normal to the disk plane, 90 means motion in the disk plane.

CSV, PNG and SVG files are written to
`results_granular_analysis/alignment/alignment_<mode>_<counter>.*`.
CSV particle numbers are one-based input rows, not physical IDs. Stationary
particles and nonunique longest axes (prolate mode) have undefined angles and
are excluded from the histogram. `--alignment-min-speed VALUE` additionally
excludes speeds at or below VALUE, in simulation velocity units.

The alignment routine also writes `_diagnostics` and `_spatial` PNG/SVG plots:
count-, speed- and kinetic-energy-weighted angular distributions, an isotropic
3D reference, absolute-cosine distribution, linear-speed/angle density diagram,
axis/plane velocity components, and spatial cluster projections (xy, xz, yz).
Energy weights require masses from `input_ellipsoid_data.dat`, matched by input
row; missing input omits that curve with a warning. Weights are normalized over
valid particles only. For oblates, axis speed is normal to the disk and plane
speed is in the disk plane.

DBSCAN clusters standardized angle in degrees and linear speed with equal
feature weight. Defaults are `--cluster-eps 0.3 --cluster-min-samples 20`.
Use `--alignment-min-speed` to exclude numerically insignificant motion.
CSV labels: -2 excluded, -1 noise, nonnegative values are clusters. The JSON
sidecar records parameters and fitted scaling; `_clusters.csv` contains group
counts and mean angles/speeds. Clusters are exploratory, parameter-dependent,
and local to a frame (labels do not track groups across time). Spatial position
is visualized but is not a clustering feature. No fixed cluster count is forced.
Compare several eps values before interpreting groups physically. The method
uses scikit-learn's DBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

The `energy` routine also writes `energy_ratio.png` and the CSV column
`translational_rotational_ratio`: summed translational energy divided by summed
rotational energy, over normalized time. This is a ratio of sums, not an average
of particle ratios. Zero or nonfinite rotational energy gives NaN (a plot gap);
small positive denominators are not clipped. The dashed line at 1 marks equal energies.

### Glättung der Energieplots

Die Standardplots der Gesamtenergie, Energiekomponenten und des
Translations-/Rotationsverhältnisses zeigen einen zentrierten gleitenden
Mittelwert über 9 Ausgabepunkte. Die Glättung wird im Plot angegeben.
CSV-Dateien und Energieberechnung bleiben unverändert. Nichtendliche Werte
unterbrechen die Kurve; über solche Lücken wird nicht gemittelt. An den
Rändern wird das Fenster verkürzt. Das Fenster zählt Ausgabepunkte, keine
physikalische Zeit (relevant bei adaptiven Zeitschritten). Die Plotfunktionen
akzeptieren `smoothing_window=1` zum Abschalten oder eine andere positive
ungerade Fenstergröße. Vorhandene Bilder werden beim nächsten Energie-Aufruf
neu erzeugt.

Energiekomponenten werden in drei Teilplots mit gemeinsamer Zeitachse und
separaten, ausdrücklich gekennzeichneten vertikalen Skalen dargestellt.
Auch der Case-Vergleich glättet die dargestellten Kurven (9 Punkte); seine
Legende steht unterhalb der Zeichenfläche. Vergleichs-CSV bleiben Rohdaten.

Einzelne Energiezeitreihen werden nach der Glättung auf maximal 300 Punkte
pro lückenfreiem Abschnitt reduziert: benachbarte Samples werden gemittelt,
Anfangs- und Endzeitpunkt bleiben erhalten. Das reduziert visuelle Unruhe,
kann aber kurze Spitzen abschwächen. Rohdaten bleiben vollständig in CSV.
`max_points=None` deaktiviert diese Reduktion in den Plotfunktionen.

## Particle rendering with a matching fluid slice

Both new analyses are part of the existing `granular-analyze` entry point:
`--only particle_fluid` and `--only porosity_time`. They are opt-in, so the
standard pipeline selection remains unchanged. Install the optional renderer
from the package directory with `python -m pip install -e ".[render]".
Porosity-time analysis does not require the renderer dependencies.

```bash
granular-analyze --only particle_fluid \
  --path ../oblaten/new-prime-gran-coll-2000_NewInput \
  --counter 3241 --z 2.25 --fluid magnitude \
  --xlim 0 50 --ylim 0 30 --fluid-clim 0 0.55 \
  --out ../bilder/generiert/oblaten_frontlicht
```

The defaults for this pipeline are **x = [0, 50], y = [0, 30]**, applied
identically to the particle rendering, exact slice panel, and overlaid figure.
They can be overridden with `--xlim` and `--ylim`. The camera looks from +z
along -z. A single non-positional light illuminates the particles from the
same +z direction; PyVista's default side lights are removed. The colour bar
for velocity magnitude is labelled `||u_f||` with double norm bars.
The default slice is z = 2.25 and the default field is velocity magnitude.
Other fields use the existing `--fluid` options.

Particle and fluid records always have exactly matching stored times.
`--time T` selects the nearest available common time; `--counter N` selects
that particle counter and requires its exact fluid match. Without either,
the latest common frame is used. The two selectors are mutually exclusive.

Files under `<out>/particle_fluid/`:

- `particle_fluid_*_kombiniert.png/.pdf`: full-depth particle projection over
  the selected 2D fluid field.
- `particle_fluid_*_vergleich.png/.pdf`: rendering and exact particle/fluid
  cross-section with the same spatial limits.
- `particle_fluid_*.json`: actual time, counters, bounds, colour limits and
  explicit light direction.

The overlaid projection can obscure fluid outside the exact particle
cross-sections; use the comparison panel to interpret the actual slice.
The figures are regenerated on each requested render. Fixed particles remain
visible. Coordinates follow the existing D_EQ=1 convention; velocity and
vortex criteria retain simulation units, while pressure is normalized.
Use **the same `--fluid-clim MIN MAX` for every case** to enforce a common
colour scale in both the combined figure and the exact-slice comparison.
For the generated Oblaten and Prolaten-Uncut examples the common limits are
`--fluid-clim 0 0.55`, covering both maxima in x=[0,50], y=[0,30]. No field is
rescaled individually when this option is given; the explicit limits are
recorded in JSON. Without this option, limits are chosen from each field.
The standalone module accepts `--clim MIN MAX`; the Python function accepts
`clim=(minimum, maximum)`.

The reusable function is
`granular_analysis.scripts.render_particle_fluid.run_particle_fluid(case, output, ...)`.
Its default `bounds` are `(0, 50, 0, 30)`; `bounds=None` retains the optional
particle-region zoom. The standalone module remains supported with `--bounds`
or `--auto-zoom`, and its `--out` is the exact target folder. `Nell_fixed` from
`input.dat` (overridable using `--skip-points` in granular-analyze or
`--skip-first` in the standalone module) determines the particles used only
for automatic zoom, not which particles are rendered.

## Porosity evolution over time

`porosity_time` evaluates the existing periodic alpha-shape slice porosity
`phi(z,t) = 1 - A_ell_union(z,t) / A_alpha(z,t)` at multiple stored times.
It produces up to **15 z profiles coloured by physical time**, plus a time
series of the **spatial mean with a ±1 standard-deviation band over z**.
It is opt-in and does not run as part of the default pipelines.

```bash
granular-analyze --only porosity_time \
  --path /path/to/case --out /path/to/analysis-output \
  --porosity-frames 15 --porosity-profiles 15 \
  --lz 5 --porosity-z-step 0.1

# Denser mean/std time series, retaining only 15 curves in the profile figure:
granular-analyze --only porosity_time --path /path/to/case \
  --porosity-frames 60 --porosity-profiles 15 \
  --time-start 50 --time-end 200 --lz 5 --porosity-z-step 0.1
```

`--porosity-frames 0` evaluates all available frames (potentially expensive).
Time targets are evenly spaced in physical time, mapped to the nearest unique
stored frames within the requested interval; no interpolation or extrapolation
is performed. Sparse data can yield fewer curves. Duplicate timestamps retain
the higher frame counter. Binary headers, not potentially stale restart tables,
define the actual time. `--counter` and `--time` are rejected for this pipeline;
use `--time-start` and `--time-end` instead. Stored time is not normalized.

Set `--lz` to the **actual periodic domain length** (default 5). The z grid is
uniform on `[0, Lz)`, with `ceil(Lz / requested_step)` planes; its actual spacing
is recorded in metadata. The final plot connects the curve back to its first
value at Lz, but that periodic endpoint is excluded from statistics.
Fixed particles are excluded using `Nell_fixed` in `input.dat`; override with
`--skip-points`. If that input parameter is absent, the existing package skip
configuration applies. `--alpha` defaults to the existing `ALPHA_2D` (0.6).

For each complete profile with N equally spaced planes:

- `porosity_mean = sum(phi_i) / N`.
- `porosity_std = sqrt(sum((phi_i - porosity_mean)^2) / N)` (`ddof=0`).

This is spatial spread over z at each time, **not** an uncertainty estimate,
standard error, variation between realizations, temporal standard deviation,
or a volume-weighted 3D porosity. The existing alpha-shape approximation and
convex-hull fallback remain in use. Undefined/nonfinite porosity, nonpositive
alpha area, or porosity outside `[0,1]` produces a missing slice. Incomplete
profiles have NaN mean/std rather than changing the averaging region silently;
`n_valid`, `n_z`, and `complete` in the CSV identify these cases.

Files under `<out>/porosity_time/`:

- `porosity_profiles.png/.pdf/.svg`: z profiles, with a time colour bar.
- `porosity_mean_std.png/.pdf/.svg`: mean and ± spatial standard deviation.
- `porosity_profiles.csv`: all sampled times and planes, areas and validity.
- `porosity_time_summary.csv`: mean, std and number of valid slices per time.
- `porosity_time.json`: case, counters, actual times, grid, alpha and definitions.
- `cache/`: per-frame profiles keyed by source path/stat and geometry settings.

`--force` recomputes the profiles; changing only the number of displayed curves
reuses valid cached profiles. No fluid checkpoints or particle render dependencies
are needed. The Python function returns `(profiles, summary)` DataFrames:

```python
from granular_analysis.scripts.run_porosity_time import run_porosity_time
profiles, summary = run_porosity_time(
    "/path/to/case", "/path/to/output", n_frames=15, n_profiles=15,
    lz=5, z_step=0.1, alpha=0.6,
)
```

A standalone CLI uses `python -m granular_analysis.scripts.run_porosity_time`
with `--path`, `--out`, `--frames`, `--profiles`, `--z-step`, `--lz`, `--alpha`,
`--skip-points`, `--time-start`, `--time-end` and `--force`. Its `--out` is the
exact output directory, without the extra `porosity_time` subdirectory.

### P0–P2 collapse products

Completed collapse histories can be analyzed together with the reusable
`analysis/collapse.py` routines. Outputs go to the specified new directory;
simulation inputs and raw results are read-only.

```bash
PYTHONPATH=granular_analysis/src python -m granular_analysis.scripts.run_collapse_analysis \
  --case Oblat=/path/to/oblate-case \
  --case Prolat=/path/to/prolate-case/results_complete \
  --out /path/to/new-delivery
```

After installing this package, the same command is available as
`granular-collapse`. Each case produces geometry/energy histories, final
geometry with a separate quasistatic validity flag, front-t95 collapse time,
alpha sensitivity, AIC/BIC/AICc surface models, binned slice porosity,
orientation and active collision-impulse statistics, initial top-layer sinking,
and same-height ambient-referenced pressure where exported fields are available.
Time follows the assignment's `t*u_s/D_eq` convention (`u_s=0.3`), and energies
are normalized by `rho_p*g*D_eq**4`.

The default rotational share threshold is 0.5; use
`--rotation-share-threshold 1` for raw rotational energies.
`--frames`, `--z-slices`, `--alpha`, `--energy-fraction`, `--hold-time` and
`--workers` control the analysis. `--reuse-energy` checks options and stored
frame times before reusing this delivery's energy products.
`--plots-only` regenerates comparisons without reading simulation histories.

`--grid-pair FORM coarse/endzustand.csv fine/endzustand.csv` supplies a matched
two-grid comparison; different particle forms cannot act as a grid pair.
`--literature references.csv` requires sourced coefficients and validity ranges
(columns `label,coefficient,exponent,chi_min,chi_max,metric,source_url`). A local
`literatur_referenzen.csv` in the delivery directory is picked up automatically.
Missing fluid/work budget terms remain unavailable; the particle mechanical
residual is explicitly not labeled dissipation. Missing collision exports
remain unknown rather than zero. Fonts follow the central pdfLaTeX/T1 Computer Modern style; SVG glyphs
are paths, and PDF fonts are embedded.

The default geometry product samples 81 stored frames per case (denser early
sampling); energy covers all native TRN times. `--frames 0` requests geometry
for every stored particle frame. The current delivery documents the exact
sample counters and the masked Uncut energy frames in its inventories.

Several runs of the same particle form can be compared with reusable P0–P2
products. This creates a separate delivery, copies existing products, checks
matching analysis options, and distinguishes the shared time interval from
each run's final snapshot:

```bash
PYTHONPATH=granular_analysis/src python -m granular_analysis.scripts.compare_collapse_cases \
  --case primeCollNick=prolaten/primeCollNick \
  --products prolat_uncut=bilder/generiert/p0_p2_oblat_prolat_uncut_20261002/prolat_uncut \
  --out bilder/generiert/vergleich_primeCollNick_prolat_uncut_NEW
```

`--case` and `--products` are repeatable; labels must be unique. `--form` defaults
to `Prolat`. Existing output product folders are protected from overwriting.
The comparison exports parameter differences, geometry, energy (raw and
filtered rotation), porosity, pressure, orientation and contact tables plus
PDF/SVG plots. Different packings and simulation settings must be considered
when interpreting differences; matching grids alone do not isolate a cause.

`granular-analyze --only renderview --xlim 0 30 --ylim 0 15` applies these
coordinate bounds to all four 3D views, keeping the configured z bounds.
Changing the bounds regenerates existing renders.

## Input-particle shapes and Zingg diagram

```bash
granular-analyze --path /path/to/case --only particle_shape
```

This input-only pipeline writes `particle_shape/particle_shape.csv` and
`particle_shape/zingg.svg` under the analysis output directory (`--out` to
choose a different directory). No TRN files are needed. Numeric rows of `input_ellipsoid_data.dat` after the selected particle skip
are included, including rows beyond `Nell`. CSV rows retain the input line number and particle ID.

The semi-axes are sorted for each particle into L ≥ I ≥ S. The plot uses
x=I/L and y=S/I with boundaries at 2/3; CSV classes are equant, disk, rod,
and blade. Axes and density must be finite and strictly positive; malformed
numeric particle rows raise an error instead of being silently skipped.
Points can overlap when several particles have the same axis ratios.

The CSV includes Corey shape factor `Sf = S / sqrt(L*I)`, equivalently
`c / sqrt(a*b)` for sorted axes a ≥ b ≥ c. Original a/b/c input columns are
preserved; sorted L/I/S columns define the calculation even if the input
axes use another order. Sf is dimensionless, size-independent, and equals
1 for spheres. The scatter points are colored by Sf on a fixed 0–1 scale.

The reusable function is
`granular_analysis.analysis.particle_shape.compute_corey_shape_factor(axes)`,
where `axes` is an N×3 array of semi-axes.

The same CSV also includes `wadell_sphericity`: the area of a sphere with
the same volume divided by the smooth ellipsoid's surface area. Surface
area uses the numerically evaluated Carlson elliptic integral
([DLMF 19.33.1](https://dlmf.nist.gov/19.33.E1)), rather than an empirical
surface-area approximation. This assumes an ideal ellipsoid; mesh
roughness is not included. `zingg_wadell.svg` provides the additional
Wadell-colored scatter plot. Both metrics equal 1 for spheres.

Reusable function: `compute_wadell_sphericity(axes)` in
`granular_analysis.analysis.particle_shape`. Both functions accept N×3
semi-axis arrays and are invariant to axis order and overall size.


### Sieblinie und Partikelausschluss

```bash
granular-analyze --path /path/to/case --only particle_shape --skip-particles 2000
```

`--skip-particles` und `--skip-points` sind Aliase. Die ersten N numerischen
Partikelzeilen werden für sämtliche Formfaktoren, Zingg-Plots und die
Sieblinie ausgeschlossen; IDs und ursprüngliche Zeilennummern bleiben
 erhalten. Ohne explizite Option gilt die bestehende Konfiguration
`USE_PARTICLE_SKIP` / `SKIP_FIRST_PARTICLES` (aktuell 2000).
`--skip-particles 0` schließt alle Input-Partikel ein. Negative Werte oder
ein Ausschluss aller Partikel führen zu einem Fehler.

Zusätzlich entstehen `sieve_curve.csv` und `sieve_curve.svg`: kumulierter
Anteil für D ≤ d, gewichtet nach Volumen, Masse und Anzahl, jeweils in
Prozent. D = 2*(a*b*c)^(1/3) ist der volumenäquivalente Kugeldurchmesser
in Input-Längeneinheiten und steht auch in `particle_shape.csv` als `d_eq`.
Gleiche Durchmesser werden gemeinsam kumuliert. Der Plot verwendet eine
logarithmische Durchmesserachse. Bei gleicher Dichte fallen Massen- und
Volumenlinie zusammen. Dies ist eine geometrische Größenverteilung;
der tatsächliche Siebdurchgang nichtkugeliger Partikel hängt zusätzlich
von Orientierung und Sieböffnung ab und wird hier nicht simuliert.

### Sieblinie mit DIN-Korngrößenklassen

`sieve_curve.svg` wird standardmäßig mit **0.615 mm pro Input-Längeneinheit**
im klassischen
logarithmischen Korngrößendiagramm nach DIN EN ISO 14688-1 dargestellt:
Ton, Schluff, Sand, Kies, Steine und Blöcke; Schluff/Sand/Kies jeweils
mit fein/mittel/grob. Die Klassengrenzen sind 0.002, 0.0063, 0.02,
0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200 und 630 mm. Diese Stützstellen
sind Klassengrenzen, keine vorgeschriebene Labor-Siebsatzliste.

`--length-unit-mm FAKTOR` überschreibt den konfigurierten Standard
`PARTICLE_LENGTH_UNIT_MM = 0.615`. `FAKTOR` bezeichnet Millimeter pro
Input-Längeneinheit. Beispiel **nur bei
1 Input-Einheit = 0.5 mm**:

```bash
granular-analyze --path /path/to/case --only particle_shape --skip-particles 2000 --length-unit-mm 0.5
```

`sieve_curve_din.csv` enthält die kumulierten Anteile an den Klassengrenzen;
`grain_classes_din.csv` die Anteile jeder Unterklasse. Grenzpartikel gehören
zur Klasse mit dieser oberen Grenze (z.B. 2 mm zum Grobsand); die letzte
Klasse ist nach oben offen. Alle Klassenanteile summieren sich zu 100 %.
Der Durchmesser bleibt volumenäquivalent; eine normgerechte experimentelle
Siebanalyse wird nicht behauptet. Ohne expliziten Millimeterfaktor gilt der bestätigte Wert 0.615 mm;
DIN-Klassen werden immer ausgegeben. Nominales D_eq=1 entspricht damit
0.615 mm; D=0.7–1.3 entspricht 0.4305–0.7995 mm.

Normbezug: [DIN EN ISO 14688-1:2020-11](https://www.dinmedia.de/de/norm/din-en-iso-14688-1/327201049).
Die Klassengrenzen sind auch in [Tabelle 1 dieser Untersuchung zur
EN ISO 14688-2](https://www.mdpi.com/2076-3417/13/21/11964) aufgeführt.


## Document-synchronous plot exports for granular-analyze

All Matplotlib plot routines use `plotting/plotstyle.py`: LaTeX typesets the
labels with T1 Computer Modern at 10 pt. Legends have no frame; plot titles
are omitted, decimal numbers use commas. Shape comparisons with identifiable
case labels use Kugel `#000000` (circle), Prolat `#000099` (horizontal ellipse),
and Oblat `#cc6600` (vertical ellipse). Cluster/time/fluid colour scales retain
their meaning. Exact 2D particle outlines are 0.5 pt and speed colours run from
white at zero to red at the highest finite 3D speed in the slice.

Every plot export writes matching `.pdf` and `.svg` files at the same stem.
Existing PNG outputs remain available. Figures are created at their final
print size, with full width exactly 160 mm (`160/25.4` inches), half width 80 mm.
The exporter keeps the physical canvas, overriding tight cropping to prevent
outside legends from enlarging the page. Include the PDF without scaling:

```latex
\includegraphics{plot_name.pdf}
```

SVG lettering consists of vector paths, not editable text. Shaded PyVista
renderings and explicitly rasterized fluid fields retain embedded raster
content inside PDF/SVG; the format does not turn a screenshot into vectors.

A working `latex` and `pdflatex` installation must be available. MiKTeX or
TeX Live needs `type1cm`, `cm-super`, `underscore` and `geometry`; PNG previews
also require `dvipng`. The current workspace has a portable TeX Live install
under `granular_analysis/.tools/texlive`; the central style finds it
automatically if system TeX is missing. Set `GRANULAR_TEX_BIN` to the TeX bin
folder for a custom installation. Missing tools produce a clear error; the
exporter does not substitute another font.

Use `--force` when regenerating existing analysis products. The energy cache
also includes the plot-style revision and requires both vector files.
Acceptance examples, portable reproduction scripts and the unscaled pdfLaTeX
test page are in `codex_lieferung/svg_plots_091026/`. These examples exercise
plot routines; native CSV-based thesis plots remain with LaTeX.

Slice legends and colour bars are horizontal above the slice. Particle and
fluid velocity plots show dimensionless speeds/components divided by
`sqrt(g D_eq)`, with `g=0.06`, `D_eq=1`. Magnitudes use double norm bars.
Velocity `--fluid-clim` bounds are in these dimensionless units; pressure and
vortex criteria retain their existing definitions. Source arrays/CSVs are
not renormalized in place.

Fluid-`magnitude`: Inferno (Dunkelviolett → Orange → Hellgelb), zentral als
`FLUID_MAGNITUDE_CMAP` definiert; die Partikelgeschwindigkeit bleibt weiß–rot.

Schnittlayout: jeweils 2,88 pt Abstand zwischen den beschrifteten Legenden
und zwischen unterster Legende und Schnitt. Y-Ticks alle 5. Automatische
Fluid-magnitude-Farbgrenzen: tatsächliches Minimum/Maximum im sichtbaren
Ausschnitt; explizite --fluid-clim-Grenzen bleiben ein manueller Override.
25 betroffene Plottests bestanden; gleiche Abstände im gerenderten Plot gemessen.

Legendentitel stehen unter den Farbskalen; Farbskalen bleiben über dem Schnitt.
Die sichtbaren Abstände sind weiterhin jeweils 2,88 pt.

Q ist in den Fluidplots dimensionslos: `Q_star = Q D_eq/g = Q/0.06`,
Beschriftung `Q D_eq/g`. Q-`--fluid-clim` wird ebenfalls dimensionslos
interpretiert; der Loader und Rohdaten bleiben unverändert.


### Fluid-Simulationsgitter im Schnitt

`--fluid-grid` blendet die tatsächlichen Zellgrenzen aus `results/fluid/x.dat`
und `y.dat` ein. Zellflächen werden aus der zweiten Spalte gelesen, nicht aus
Achsenticks oder einer angenommenen konstanten Gitterweite. Die Darstellung
liegt hinter den opaken Partikelschnitten; das normale Achsengitter wird im
exakten Schnitt ausgeblendet. Fluidwerte werden bei aktiviertem Gitter auf
den tatsächlichen Zellflächen gezeichnet, auch bei ungleichmäßiger Gitterweite.

```bash
granular-analyze --only 2d --path /pfad/zum/fall \
  --fluid magnitude --fluid-grid --time 90 --z 2.25 \
  --xlim 25 30 --ylim 0 5
```

Ohne explizites `--fluid` verwendet `--fluid-grid` magnitude. Die Option ist
für `2d`, `slices` und `particle_fluid` verfügbar; die exakte Partikel-/Fluidzeit
muss wie bei anderen Fluidplots übereinstimmen. Dateinamen bekommen `_grid`,
um bestehende Ausgaben ohne Fluidgitter zu erhalten. SVG/PDF werden wie gewohnt
gemeinsam ausgegeben. `--xlim`/`--ylim` erlauben einen lesbaren Gitter-Zoom.

Horizontale Farbskalen/Legenden werden nach der Layoutberechnung links und
rechts bündig zur tatsächlichen Schnittfläche ausgerichtet, auch bei Zooms.
Die Ausrichtung bleibt im PDF/SVG-Export erhalten.

Im Modus `--fluid-grid`: Fluid-Zellgrenzen weiß, x-/y-Ticks alle 2,5.
Farbskalen sind bündig zur Schnittbreite ausgerichtet.

Partikelgeschwindigkeit: zehn diskrete, gleich breite Geschwindigkeitsklassen
zwischen null und dem maximalen normierten Betrag im Schnitt, von Weiß bis Rot.
Partikelfüllung und Farbskala verwenden dieselben Klassengrenzen.


### Particle velocity direction arrows

Add `--velocity-arrows` to a `--only 2d` or `--only slices` run. Arrows are
black (fill and outline), use the in-plane `(u,v)` direction, and are scaled
with particle size. Speed remains represented by the ten-step white/red
colour scale. Zero in-plane velocities and invalid velocities have no arrow.
Use `--no-velocity-colors` together with `--velocity-arrows` for arrows only.
Outputs with arrows have an `_arrows` suffix.

```bash
granular-analyze --path . --only 2d --fluid magnitude --z 2.25 --velocity-arrows
```
