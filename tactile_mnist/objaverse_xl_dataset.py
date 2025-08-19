from __future__ import annotations

import hashlib
import logging
import sys
import traceback
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import datasets
import loguru
import objaverse.xl
import pandas as pd
import requests
import trimesh
from trimesh import Trimesh
from trimesh.exchange.load import mesh_loaders

from tactile_mnist import MeshDataset
from .huggingface_dataset import DataPointType, HuggingfaceDatapointField
from .mesh_dataset import MeshDataPoint

logger = logging.getLogger(__name__)

loguru.logger.remove()
loguru.logger.add(sys.stdout, level="WARNING")


class ErrorHandler:
    def __init__(self):
        self.called = False

    def __call__(self, *args, **kwargs):
        self.called = True


class ObjaverseXLError(RuntimeError):
    pass


class ObjaverseXLMeshError(ObjaverseXLError):
    pass


class ObjaverseXLDownloaderError(ObjaverseXLError):
    pass


@lru_cache(maxsize=1)
def get_fallback_mesh() -> trimesh.Trimesh:
    return trimesh.load_mesh(files("tactile_mnist.resources").joinpath("fallback.obj"))


def load_objaverse_xl_mesh(d: dict) -> Trimesh:
    download_dir = Path.home() / ".cache" / "tactile-mnist" / "objaverse_xl"
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        if d["source"] == "github":
            github_download_dir = download_dir / "github"
            github_download_dir.mkdir(exist_ok=True)

            parsed_url = urlparse(d["fileIdentifier"])
            url_path = Path(parsed_url.path)
            parts = list(url_path.parts)
            parts[3] = "raw"
            parsed_url = parsed_url._replace(path=str(Path(*parts)))
            file_ending = url_path.suffix
            identifier_hash = hashlib.sha256(
                d["fileIdentifier"].encode("utf-8")
            ).hexdigest()
            filename = github_download_dir / f"{identifier_hash}{file_ending}"

            if not filename.exists():
                response = requests.get(parsed_url.geturl())
                response.raise_for_status()

                with filename.open("wb") as file:
                    file.write(response.content)
        else:
            missing_object_handler = ErrorHandler()
            modified_object_handler = ErrorHandler()

            model_files = objaverse.xl.download_objects(
                pd.DataFrame([d]),
                processes=1,
                handle_missing_object=missing_object_handler,
                handle_modified_object=modified_object_handler,
                download_dir=str(download_dir),
            )
            if len(model_files) == 0:
                if missing_object_handler.called:
                    raise ObjaverseXLDownloaderError(
                        f"Mesh not found in Objaverse XL ({d['fileIdentifier']})."
                    )
                elif modified_object_handler.called:
                    raise ObjaverseXLDownloaderError(
                        f"Objaverse XL mesh did not pass the integrity check ({d['fileIdentifier']})."
                    )
                raise ObjaverseXLDownloaderError(
                    f"Unknown error when downloading mesh from Objaverse XL ({d['fileIdentifier']})."
                )
            filename = Path(model_files[d["fileIdentifier"]])
        mesh = trimesh.load_mesh(filename)
        if mesh.volume == 0:
            raise ObjaverseXLMeshError(
                f"Mesh from Objaverse XL ({d['fileIdentifier']}) has zero volume."
            )
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh, multibody=True)
        trimesh.repair.fix_winding(mesh)
        return mesh
    except:
        logger.warning(
            f"Failed to download mesh from Objaverse XL ({d['fileIdentifier']}):\n{traceback.format_exc()}"
        )
        logger.warning("Falling back to fallback mesh.")
        return get_fallback_mesh()


class ObjaverseXLMeshDataPoint(MeshDataPoint):
    id: str = HuggingfaceDatapointField(("sha256",), lambda d: d["sha256"])
    label: int = HuggingfaceDatapointField((), lambda d: 0)
    mesh: Trimesh = HuggingfaceDatapointField(
        ("fileIdentifier", "source", "sha256"), load_objaverse_xl_mesh
    )


class ObjaverseXLMeshDataset(
    MeshDataset[ObjaverseXLMeshDataPoint, "ObjaverseXLMeshDataset"]
):
    def __init__(
        self,
        huggingface_dataset: datasets.Dataset | datasets.IterableDataset,
        cache_size: int | Literal["full"] = 0,
    ):
        # Filter out Thingiverse objects as they do not allow scripted downloads
        print("Filtering out invalid objects from Objaverse XL dataset...")
        super().__init__(
            huggingface_dataset=huggingface_dataset.filter(self.__data_point_loadable),
            cache_size=cache_size,
        )

    @staticmethod
    def __data_point_loadable(data_point: dict[str, str]) -> bool:
        if data_point["source"] == "thingiverse":
            return False
        if data_point["source"] == "github":
            return data_point["fileType"].lower() in mesh_loaders
        return True

    def _get_data_point_type(self) -> type[DataPointType]:
        return ObjaverseXLMeshDataPoint
