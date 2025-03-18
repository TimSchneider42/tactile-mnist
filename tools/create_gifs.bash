#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

ENVS=(
  TactileMNIST-v0
  Starstruck-v0
)

OUTPUT_DIR="$SCRIPT_DIR/../doc/img/env"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
for env in "${ENVS[@]}"; do
  ap-gym-create-env-gif "tactile_mnist:$env" "${OUTPUT_DIR}/$env.gif" &
done

wait
