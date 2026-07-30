#!/usr/bin/env python3
"""
Encode meshes of a mesh dataset into COD-VAE latents (Cho et al., ICCV 2025;
https://github.com/TimSchneider42/cod-vae), decode them back, and show the original
(left) and the reconstruction (right) side by side in an interactive viewer.

Requires `pip install cod-vae[hub]` in addition to the tactile-mnist environment.
"""

from __future__ import annotations

import argparse

import numpy as np
import trimesh
from datasets import load_dataset
from scipy.spatial import cKDTree

from cod_vae import CODVAE
from tactile_mnist import SimpleMeshDataset
from tactile_mnist.constants import CELL_SIZE


def chamfer_distance(
    mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, num_samples: int = 50_000
) -> float:
    """Symmetric Chamfer (mean nearest-neighbor surface-to-surface) distance in meters."""
    samples_a, _ = trimesh.sample.sample_surface(mesh_a, num_samples)
    samples_b, _ = trimesh.sample.sample_surface(mesh_b, num_samples)
    dist_ab = cKDTree(samples_b).query(samples_a, workers=-1)[0]
    dist_ba = cKDTree(samples_a).query(samples_b, workers=-1)[0]
    return float((dist_ab.mean() + dist_ba.mean()) / 2)


def place_on_platform(
    mesh: trimesh.Trimesh,
    cell_size: np.ndarray,
    rng: np.random.Generator | None = None,
) -> trimesh.Trimesh:
    """
    Center the mesh in the cell and drop it onto the platform (min z = 0). If rng is given,
    additionally apply a random Z-rotation and a random XY offset that keeps the object inside
    the cell with a 1 cm margin (mirroring the environment's pose randomization).
    """
    mesh = mesh.copy()
    if rng is not None:
        angle = rng.uniform(-np.pi, np.pi)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
    bounds_center = (mesh.bounds[0] + mesh.bounds[1]) / 2
    mesh.apply_translation([-bounds_center[0], -bounds_center[1], -mesh.bounds[0][2]])
    if rng is not None:
        margin = 0.01
        max_offset = np.maximum(
            cell_size / 2 - margin - (mesh.bounds[1, :2] - mesh.bounds[0, :2]) / 2, 0.0
        )
        mesh.apply_translation([*rng.uniform(-max_offset, max_offset), 0.0])
    return mesh


def build_side_by_side_scene(
    original: trimesh.Trimesh,
    reconstruction: trimesh.Trimesh | None,
    cell_size: np.ndarray,
    reconstruction_color: tuple[int, int, int, int] = (66, 135, 245, 255),
    gap: float = 0.02,
) -> trimesh.Scene:
    """
    Build a scene showing the original (left, gray) and the reconstruction (right) on
    separate platform cells.
    """
    offset = (cell_size[0] + gap) / 2
    platform_thickness = 0.002
    scene = trimesh.Scene()
    for name, mesh, color, sign in [
        ("original", original, [160, 160, 160, 255], -1),
        ("reconstruction", reconstruction, list(reconstruction_color), 1),
    ]:
        platform = trimesh.creation.box(
            [cell_size[0], cell_size[1], platform_thickness],
            trimesh.transformations.translation_matrix(
                [sign * offset, 0, -platform_thickness / 2]
            ),
        )
        platform.visual.face_colors = [230, 230, 230, 255]
        scene.add_geometry(platform, node_name=f"{name}_platform")
        if mesh is not None:
            mesh = mesh.copy()
            mesh.visual.face_colors = color
            mesh.apply_translation([sign * offset, 0, 0])
            scene.add_geometry(mesh, node_name=name)
    return scene


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="TimSchneider42/tactile-mnist-abc-dataset-small",
        help="Name or path of the mesh dataset to load.",
    )
    parser.add_argument("-s", "--split", type=str, default="train")
    parser.add_argument(
        "-i",
        "--indices",
        type=int,
        nargs="+",
        help="Dataset indices to encode. If not given, the entire dataset is cycled "
        "through in random order.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="TimSchneider42/cod-vae",
        help="Model to load: a Hugging Face Hub repo id, a local npz checkpoint, or a "
        "directory containing an official COD-VAE release (config.yaml + *.pt).",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "torch", "jax"),
        default="auto",
        help="Compute backend (auto prefers JAX if installed, falling back to PyTorch).",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run on (backend-specific, e.g. cuda or cpu).",
    )
    parser.add_argument(
        "--randomize-pose",
        action="store_true",
        help="Randomize the object's pose in the cell like the environment does.",
    )
    args = parser.parse_args()

    vae = CODVAE.from_pretrained(args.model, backend=args.backend, device=args.device)
    num_latent_values = vae.config.num_latents * vae.config.latent_dim
    print(
        f"Loaded COD-VAE: {vae.config.num_latents} latent vectors x "
        f"{vae.config.latent_dim} dims = {num_latent_values} values, "
        f"using the {vae.backend} backend"
    )

    cell_size = np.array(CELL_SIZE)
    dataset = SimpleMeshDataset(load_dataset(args.dataset, split=args.split))
    rng = np.random.default_rng(args.seed)
    if args.indices is not None:
        indices = args.indices
    else:
        indices = rng.permutation(len(dataset))

    for index in indices:
        dp = dataset[int(index)]
        mesh = place_on_platform(
            dp.mesh, cell_size, rng if args.randomize_pose else None
        )

        z, transform = vae.encode_mesh(mesh, seed=int(index), return_transform=True)
        print(f"\nMesh {dp.id} (index {index}):")
        print(f"  Latent: {z.shape} -> {z.size} values")
        reconstruction = vae.decode_mesh(z, transform=transform)

        if reconstruction.is_empty:
            print("  Reconstruction is empty!")
            reconstruction = None
        else:
            chamfer = chamfer_distance(mesh, reconstruction)
            print(f"  Chamfer distance: {chamfer * 1000:.2f} mm")

        scene = build_side_by_side_scene(mesh, reconstruction, cell_size)
        print(
            "  Showing original (left) and reconstruction (right); close the window to continue."
        )
        scene.show(
            caption=f"{dp.id}: original (left) vs COD-VAE reconstruction (right)"
        )


if __name__ == "__main__":
    main()
