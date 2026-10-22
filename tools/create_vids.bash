#!/bin/bash
# Creates the preview videos in docs/img/env for the registered environments. Without environment IDs, it covers every
# registered tactile_mnist environment except the -train aliases and the -sparse variants.
#   create_vids.bash [--recreate-existing] [ENV_ID ...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

RECREATE_EXISTING=false
ENVS=()
for arg in "$@"; do
  case "$arg" in
  --recreate-existing) RECREATE_EXISTING=true ;;
  -*)
    echo "Usage: $0 [--recreate-existing] [ENV_ID ...]" >&2
    exit 1
    ;;
  *) ENVS+=("$arg") ;;
  esac
done

if [ ${#ENVS[@]} -eq 0 ]; then
  mapfile -t ENVS < <(
    python -c '
import gymnasium as gym
import tactile_mnist

for env_id, spec in gym.registry.items():
    entry_point = spec.entry_point
    module = entry_point if isinstance(entry_point, str) else entry_point.__module__
    if module.startswith("tactile_mnist") and not env_id.endswith(("-train-v0", "-sparse-v0")):
        print(env_id)
'
  ) || exit 1
fi

create_vid() {
  local env="$1"
  local output="$2"
  if ! $RECREATE_EXISTING && [ -f "$output" ]; then
    echo "Skipping $output as it exists already (use --recreate-existing to recreate it)."
    return
  fi
  ap-gym-create-env-vid "tactile_mnist:$env" "$output"
}

OUTPUT_DIR="$SCRIPT_DIR/../docs/img/env"
mkdir -p "${OUTPUT_DIR}"
for env in "${ENVS[@]}"; do
  echo "Creating video for $env..."
  create_vid "$env" "${OUTPUT_DIR}/$env.webp"
done
