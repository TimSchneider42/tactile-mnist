#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

ENVS=(
  TactileMNIST-v0
  TactileMNIST-test-v0
  TactileMNIST-CycleGAN-v0
  TactileMNIST-CycleGAN-test-v0
  TactileMNIST-Depth-v0
  TactileMNIST-Depth-test-v0
  TactileMNISTVolume-v0
  TactileMNISTVolume-test-v0
  TactileMNISTVolume-CycleGAN-v0
  TactileMNISTVolume-CycleGAN-test-v0
  TactileMNISTVolume-Depth-v0
  TactileMNISTVolume-Depth-test-v0
  Starstruck-v0
  Starstruck-test-v0
  Starstruck-Depth-v0
  Starstruck-Depth-test-v0
  Toolbox-v0
  Toolbox-Depth-v0
  Minecraft-v0
  Minecraft-Depth-v0
  ABCVolume-v0
  ABCVolume-Depth-v0
  ABCCenterOfMass-v0
  ABCCenterOfMass-Depth-v0
)

OUTPUT_DIR="$SCRIPT_DIR/../doc/img/env"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
for env in "${ENVS[@]}"; do
  echo "Creating GIF for $env..."
  ap-gym-create-env-vid "tactile_mnist:$env" "${OUTPUT_DIR}/$env.gif"
done

wait
