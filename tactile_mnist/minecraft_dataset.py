"""Mesh dataset of Minecraft items, built from Mojang's official game assets.

The item textures are downloaded directly from Mojang's servers on first use and
cached locally, so no Minecraft assets are redistributed with this package. The
downloaded assets and the meshes generated from them remain the property of
Mojang and are subject to the Minecraft EULA
(https://www.minecraft.net/en-us/eula) -- in particular, do not redistribute
them.

NOT AN OFFICIAL MINECRAFT PRODUCT. NOT APPROVED BY OR ASSOCIATED WITH MOJANG OR
MICROSOFT.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import datasets
import numpy as np
import requests
from PIL import Image

VERSION_MANIFEST_URL = "https://launchermeta.mojang.com/mc/game/version_manifest_v2.json"
ITEM_TEXTURE_PREFIX = "assets/minecraft/textures/item/"
DEFAULT_MINECRAFT_VERSION = "1.21.8"

TEXTURE_RESOLUTION = 16
# 16 px x 0.00625 m/px = 0.1 m item width, extruded one pixel deep
PIXEL_SIZE = 0.00625
ALPHA_THRESHOLD = 1


def _default_cache_dir(version: str) -> Path:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return cache_home / "tactile_mnist" / "minecraft-item-textures" / version


def fetch_minecraft_item_textures(
    version: str = DEFAULT_MINECRAFT_VERSION,
    cache_dir: Path | str | None = None,
) -> dict[str, np.ndarray]:
    """Download the official Minecraft client.jar from Mojang and extract all
    16x16 item textures as RGBA arrays.

    The textures are downloaded from Mojang's servers on first use and cached
    locally, so no Minecraft assets have to be redistributed with this package.
    """
    cache_dir = (
        _default_cache_dir(version) if cache_dir is None else Path(cache_dir)
    )
    cache_file = cache_dir / "item_textures.npz"
    if cache_file.exists():
        with np.load(cache_file) as data:
            return {name: data[name] for name in data.files}

    manifest = requests.get(VERSION_MANIFEST_URL).json()
    version_meta_url = next(
        (v["url"] for v in manifest["versions"] if v["id"] == version), None
    )
    if version_meta_url is None:
        raise ValueError(f"Minecraft version {version} not found in Mojang's manifest.")
    version_meta = requests.get(version_meta_url).json()
    client_jar_url = version_meta["downloads"]["client"]["url"]
    jar_response = requests.get(client_jar_url)
    jar_response.raise_for_status()

    textures = {}
    with zipfile.ZipFile(io.BytesIO(jar_response.content)) as jar:
        for file_path in jar.namelist():
            if not (
                file_path.startswith(ITEM_TEXTURE_PREFIX)
                and file_path.endswith(".png")
            ):
                continue
            name = file_path[len(ITEM_TEXTURE_PREFIX) : -len(".png")]
            if "/" in name:
                continue
            with jar.open(file_path) as f:
                img = Image.open(f).convert("RGBA")
            if img.size != (TEXTURE_RESOLUTION, TEXTURE_RESOLUTION):
                continue
            textures[name] = np.asarray(img)

    cache_dir.mkdir(parents=True, exist_ok=True)
    # np.savez_compressed appends ".npz" to filenames not ending in it
    tmp_file = cache_file.with_name(f"{cache_file.stem}.tmp.npz")
    np.savez_compressed(tmp_file, **textures)
    tmp_file.replace(cache_file)
    return textures


def _strip_frame_number(name: str) -> str:
    # Animation frames (e.g. compass_00 ... compass_31) all describe the same item
    tokens = name.split("_")
    while len(tokens) > 1 and tokens[-1].isdigit():
        tokens = tokens[:-1]
    return "_".join(tokens)


def _common_suffix(token_lists: list[list[str]]) -> list[str]:
    n = 0
    min_len = min(len(t) for t in token_lists)
    while n < min_len and all(t[-n - 1] == token_lists[0][-n - 1] for t in token_lists):
        n += 1
    return token_lists[0][len(token_lists[0]) - n :]


def _common_prefix(token_lists: list[list[str]]) -> list[str]:
    n = 0
    min_len = min(len(t) for t in token_lists)
    while n < min_len and all(t[n] == token_lists[0][n] for t in token_lists):
        n += 1
    return token_lists[0][:n]


def _group_name(members: list[str]) -> str:
    # Collapse members to their longest common trailing tokens (wooden_axe,
    # stone_axe, ... -> axe) or leading tokens (ender_eye, ender_pearl -> ender)
    token_lists = [m.split("_") for m in members]
    for common in (_common_suffix(token_lists), _common_prefix(token_lists)):
        if common:
            return "_".join(common)
    # Otherwise, cluster members by common trailing tokens and join the cluster
    # names (mushroom_stew, suspicious_stew, beetroot_soup, bowl ->
    # stew__beetroot_soup__bowl)
    clusters: dict[str, list[list[str]]] = {}
    for tokens in token_lists:
        clusters.setdefault(tokens[-1], []).append(tokens)
    named = [
        ("_".join(_common_suffix(token_lists)), len(token_lists))
        for token_lists in clusters.values()
    ]
    return "__".join(name for name, size in sorted(named, key=lambda x: (-x[1], x[0])))


def group_textures_by_silhouette(
    textures: dict[str, np.ndarray],
) -> list[tuple[str, np.ndarray]]:
    """Deduplicate textures by their silhouette (alpha mask), as items that only
    differ in color are indistinguishable to a tactile sensor.

    Returns (group_name, representative_texture) tuples sorted by the name of
    each group's alphabetically first member.
    """
    groups: dict[bytes, list[str]] = {}
    for name in sorted(textures):
        mask = textures[name][..., 3] >= ALPHA_THRESHOLD
        if not mask.any():
            continue
        groups.setdefault(mask.tobytes(), []).append(name)

    result = []
    for members in groups.values():
        if len(members) == 1:
            name = members[0]
        else:
            # Animation frames of the same item (compass_00 ... compass_31)
            # count as a single member
            frame_stripped = sorted({_strip_frame_number(m) for m in members})
            name = (
                frame_stripped[0]
                if len(frame_stripped) == 1
                else _group_name(frame_stripped)
            )
        result.append((name, textures[members[0]]))

    seen: dict[str, int] = {}
    deduped_names = []
    for name, _ in result:
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        deduped_names.append(name)
    return [(name, texture) for name, (_, texture) in zip(deduped_names, result)]


def texture_to_mesh(
    rgba: np.ndarray,
    pixel_size: float = PIXEL_SIZE,
    thickness: float = PIXEL_SIZE,
) -> dict[str, np.ndarray]:
    """Extrude the opaque pixels of an RGBA texture into a 3D mesh, the same way
    Minecraft renders dropped items.

    The texture is centered on the XY origin (one pixel = pixel_size meters, image
    "up" = +y) and extruded from z = 0 to z = thickness. Each face is colored with
    the color of the pixel it belongs to. Returns a dict with "vertices", "faces"
    and "face_colors" suitable for trimesh.Trimesh(**d).
    """
    height, width = rgba.shape[:2]
    opaque = rgba[..., 3] >= ALPHA_THRESHOLD

    vertex_indices: dict[tuple[int, int, int], int] = {}
    vertices = []
    faces = []
    face_colors = []

    def vertex(col: int, row: int, top: bool) -> int:
        key = (col, row, top)
        if key not in vertex_indices:
            vertex_indices[key] = len(vertices)
            vertices.append(
                (
                    (col - width / 2) * pixel_size,
                    (height / 2 - row) * pixel_size,
                    thickness if top else 0.0,
                )
            )
        return vertex_indices[key]

    def quad(corners: list[int], color: np.ndarray):
        # Corners in counter-clockwise order as seen from outside, split along
        # the diagonal between the second and fourth corner
        faces.append((corners[1], corners[2], corners[3]))
        faces.append((corners[1], corners[3], corners[0]))
        face_colors.extend([color, color])

    for row in range(height):
        for col in range(width):
            if not opaque[row, col]:
                continue
            color = np.array([*rgba[row, col, :3], 255])
            # Corner columns/rows: left/right = col/col + 1, top/bottom = row/row + 1
            tl_t = vertex(col, row, True)
            tr_t = vertex(col + 1, row, True)
            br_t = vertex(col + 1, row + 1, True)
            bl_t = vertex(col, row + 1, True)
            tl_b = vertex(col, row, False)
            tr_b = vertex(col + 1, row, False)
            br_b = vertex(col + 1, row + 1, False)
            bl_b = vertex(col, row + 1, False)
            quad([tl_t, bl_t, br_t, tr_t], color)  # top (+z)
            quad([bl_b, tl_b, tr_b, br_b], color)  # bottom (-z)
            if row == 0 or not opaque[row - 1, col]:  # +y side
                quad([tl_b, tl_t, tr_t, tr_b], color)
            if row == height - 1 or not opaque[row + 1, col]:  # -y side
                quad([bl_t, bl_b, br_b, br_t], color)
            if col == 0 or not opaque[row, col - 1]:  # -x side
                quad([tl_b, bl_b, bl_t, tl_t], color)
            if col == width - 1 or not opaque[row, col + 1]:  # +x side
                quad([br_b, tr_b, tr_t, br_t], color)

    return {
        "vertices": np.array(vertices, dtype=np.float32),
        "faces": np.array(faces, dtype=np.int32),
        "face_colors": np.array(face_colors, dtype=np.int32),
    }


def load_minecraft_item_mesh_dataset(
    version: str = DEFAULT_MINECRAFT_VERSION,
    cache_dir: Path | str | None = None,
) -> datasets.Dataset:
    """Build a mesh dataset of Minecraft items from Mojang's official assets.

    This is a drop-in replacement for a HuggingFace mesh dataset (as consumed by
    SimpleMeshDataset), except that the underlying assets are downloaded directly
    from Mojang's servers instead of being redistributed.
    """
    textures = fetch_minecraft_item_textures(version=version, cache_dir=cache_dir)
    grouped = group_textures_by_silhouette(textures)
    meshes = [texture_to_mesh(texture) for _, texture in grouped]
    names = [name for name, _ in grouped]
    return datasets.Dataset.from_dict(
        {
            "id": names,
            "label": list(range(len(names))),
            "mesh.vertices": [m["vertices"] for m in meshes],
            "mesh.faces": [m["faces"] for m in meshes],
            "mesh.face_colors": [m["face_colors"] for m in meshes],
        },
        features=datasets.Features(
            {
                "id": datasets.Value("string"),
                "label": datasets.ClassLabel(names=names),
                "mesh.vertices": datasets.Sequence(
                    datasets.Sequence(datasets.Value("float32"), length=3)
                ),
                "mesh.faces": datasets.Sequence(
                    datasets.Sequence(datasets.Value("int32"), length=3)
                ),
                "mesh.face_colors": datasets.Sequence(
                    datasets.Sequence(datasets.Value("int32"), length=4)
                ),
            }
        ),
    )
