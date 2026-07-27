#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

RECREATE_EXISTING=false
for arg in "$@"; do
  case "$arg" in
  --recreate-existing) RECREATE_EXISTING=true ;;
  *)
    echo "Usage: $0 [--recreate-existing]" >&2
    exit 1
    ;;
  esac
done

ENVS=(
  TactileMNIST-v0
  TactileMNIST-test-v0
  TactileMNIST-CycleGAN-v0
  TactileMNIST-CycleGAN-test-v0
  TactileMNIST-Depth-v0
  TactileMNIST-Depth-test-v0
  TactileMNISTSnap-v0
  TactileMNISTSnap-test-v0
  TactileMNISTSnap-CycleGAN-v0
  TactileMNISTSnap-CycleGAN-test-v0
  TactileMNISTSnap-Depth-v0
  TactileMNISTSnap-Depth-test-v0
  TactileMNISTRealSnap-v0
  TactileMNISTRealSnap-test-v0
  TactileMNISTVolume-v0
  TactileMNISTVolume-test-v0
  TactileMNISTVolume-CycleGAN-v0
  TactileMNISTVolume-CycleGAN-test-v0
  TactileMNISTVolume-Depth-v0
  TactileMNISTVolume-Depth-test-v0
  TactileMNISTVolumeSnap-v0
  TactileMNISTVolumeSnap-test-v0
  TactileMNISTVolumeSnap-CycleGAN-v0
  TactileMNISTVolumeSnap-CycleGAN-test-v0
  TactileMNISTVolumeSnap-Depth-v0
  TactileMNISTVolumeSnap-Depth-test-v0
  TactileMNISTVolumeRealSnap-v0
  TactileMNISTVolumeRealSnap-test-v0
  TactileMNISTCenterOfMass-v0
  TactileMNISTCenterOfMass-test-v0
  TactileMNISTCenterOfMass-Depth-v0
  TactileMNISTCenterOfMass-Depth-test-v0
  TactileMNISTCenterOfMassSnap-v0
  TactileMNISTCenterOfMassSnap-test-v0
  TactileMNISTCenterOfMassSnap-Depth-v0
  TactileMNISTCenterOfMassSnap-Depth-test-v0
  Starstruck-v0
  Starstruck-test-v0
  Starstruck-Depth-v0
  Starstruck-Depth-test-v0
  Toolbox-v0
  Toolbox-Depth-v0
  Minecraft-v0
  Minecraft-Depth-v0
  MinecraftPose-v0
  MinecraftPose-Depth-v0
  MinecraftShape-v0
  MinecraftShape-Depth-v0
  ABCVolume-v0
  ABCVolume-test-v0
  ABCVolume-Depth-v0
  ABCVolume-Depth-test-v0
  ABCCenterOfMass-v0
  ABCCenterOfMass-test-v0
  ABCCenterOfMass-Depth-v0
  ABCCenterOfMass-Depth-test-v0
  TactileMNISTShape-v0
  TactileMNISTShape-test-v0
  TactileMNISTShape-CycleGAN-v0
  TactileMNISTShape-CycleGAN-test-v0
  TactileMNISTShape-Depth-v0
  TactileMNISTShape-Depth-test-v0
  TactileMNISTShapeSnap-v0
  TactileMNISTShapeSnap-test-v0
  TactileMNISTShapeSnap-CycleGAN-v0
  TactileMNISTShapeSnap-CycleGAN-test-v0
  TactileMNISTShapeSnap-Depth-v0
  TactileMNISTShapeSnap-Depth-test-v0
  ABCShape-v0
  ABCShape-test-v0
  ABCShape-Depth-v0
  ABCShape-Depth-test-v0
)

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

wait
