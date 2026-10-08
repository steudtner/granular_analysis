#!/usr/bin/env bash
set -euo pipefail

case_path="${1:-.}"
output_path="${2:-}"
python_bin="${PYTHON:-python}"

args=(--path "$case_path" --only 2d slope --force --alphashape --upper-edge)
if [[ -n "$output_path" ]]; then
    args+=(--out "$output_path")
fi

exec "$python_bin" -m granular_analysis.scripts.run_all "${args[@]}"
