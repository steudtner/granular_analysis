# Slope Analysis: R and Python Comparison

## Purpose

The existing slope workflow uses R's `segmented` package to fit continuous
piecewise-linear models to the upper edge of the granular material. An
experimental Python implementation was added to check whether the same result
can be obtained without changing the production R workflow.

The comparison used the same real upper-edge profile for both methods:

```text
/data/horse/ws/s4713238-DA/results_granular_analysis/alphashape_last_timestep.csv
```

The profile contains 35 `(x, z_max)` points.

## Methods

- **R:** linear regression followed by 1-, 2-, and 3-breakpoint fits using
  `segmented`.
- **Python:** continuous linear splines fitted with NumPy least squares and
  SciPy breakpoint optimization.
- Both methods report breakpoint positions, segment slopes, slope angles,
  AIC, and BIC.

## Results

| Breakpoints | Method | Breakpoint positions | Segment slopes | AIC | BIC |
| ---: | --- | --- | --- | ---: | ---: |
| 0 | R | - | `-0.2799` | 89.945 | 94.611 |
| 0 | Python | - | `-0.2799` | 89.945 | 94.611 |
| 1 | R | `17.3241` | `-0.1722; -0.3892` | 70.123 | 77.900 |
| 1 | Python | `17.3240` | `-0.1722; -0.3892` | 70.123 | 77.900 |
| 2 | R | `19.4756; 23.1435` | `-0.1663; -0.6486; -0.3366` | 70.878 | 81.765 |
| 2 | Python | `14.6766; 16.5047` | `-0.2453; 0.4611; -0.4052` | 68.022 | 78.910 |
| 3 | R | `4.7725; 11.0317; 16.5484` | `0.1659; -0.5254; 0.1267; -0.3987` | 58.470 | 72.468 |
| 3 | Python | `6.4230; 13.2071; 15.9214` | `0.0536; -0.6810; 0.9653; -0.4052` | 54.041 | 68.039 |

For the one-breakpoint model, the corresponding angles are approximately
`-9.77 degrees` and `-21.27 degrees` for both implementations.

## Interpretation

The linear and one-breakpoint models agree essentially exactly. This confirms
that the Python implementation reproduces the central R result.

The methods find different solutions when two or three breakpoints are
allowed. The Python fits have lower in-sample AIC and BIC, but contain sharp
positive and negative slope reversals. These additional segments appear more
likely to describe local surface variation than distinct physical pile
regions. The difference also demonstrates that the higher-complexity fit is
sensitive to the optimization method and local minima.

## Recommendation

Keep R's `segmented` implementation as the production slope analysis. For this
profile, use the one-breakpoint model as the most stable and interpretable
result:

- breakpoint: `x = 17.324`
- first slope: `-0.1722` (`-9.77 degrees`)
- second slope: `-0.3892` (`-21.27 degrees`)

Use the Python method as an independent comparison and sensitivity check. A
future extension could add bootstrap confidence intervals and minimum segment
length constraints before interpreting models with multiple breakpoints.

## Reproducing the comparison

Load R on Romeo, activate the Python environment, and install the package in
editable mode:

```bash
module load release/2026
module load GCC/14.2.0
module load R/4.5.1
source /data/horse/ws/s4713238-DA/envRomeo/bin/activate

cd /data/horse/ws/s4713238-DA/granular_analysis
python -m pip install -e .
```

Then generate or select the upper-edge data and run the comparison:

```bash
cd /path/to/case
granular-analyze --only 2d
granular-compare-slopes
```

To compare an existing CSV explicitly:

```bash
granular-compare-slopes \
  --csv /path/to/alphashape_last_timestep.csv \
  --out /path/to/slope-comparison \
  --max-breakpoints 3
```

The output contains:

- `slope_results_R.csv`
- `slope_results_python.csv`
- `slope_method_comparison.csv`
- separate `r/` and `python/` plot directories

The normal `granular-analyze --only slope` pipeline continues to use R.
