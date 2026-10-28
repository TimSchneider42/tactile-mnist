#!/bin/bash
# Assemble the project page in <out dir> (default _site): docs/ plus data.json, the baseline curves of this repository
# and of ap_gym (its baseline-data/, from a checkout of active-perception-gym) for the cards' bar charts.
# The Pages workflow runs this on main; locally, `docs/tools/build_site.sh ../active-perception-gym docs` writes
# docs/data.json (ignored by git) for a preview.
# Usage: docs/tools/build_site.sh <ap_gym checkout> [out dir]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
AP_GYM=${1:?usage: build_site.sh <ap_gym checkout> [out dir]}
OUT=${2:-$ROOT/_site}
[ -d "$AP_GYM/baseline-data" ] || { echo "no baseline-data/ in $AP_GYM" >&2; exit 1; }
mkdir -p "$OUT"
if [ "$(realpath "$OUT")" != "$ROOT/docs" ]; then
  rm -rf "$OUT"
  mkdir -p "$OUT"
  cp -r "$ROOT/docs/." "$OUT/"
fi
python3 "$ROOT/docs/tools/make_data_json.py" "$OUT/data.json" "$ROOT/baseline-data" "$AP_GYM/baseline-data"
