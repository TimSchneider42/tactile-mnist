from __future__ import annotations

import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import filelock
import tqdm

logger = logging.getLogger(__name__)

DEFAULT_RESOURCE_DIR = Path.home() / ".local" / "share" / "tactile-mnist"
DEFAULT_BASE_URL = "https://archimedes.ias.informatik.tu-darmstadt.de/s/EiFPmyqa34DLF8S"

custom_remote_search_locations = list(
    map(Path, os.environ.get("TACTILE_MNIST_REMOTE_DATA_PATH", "").split(":"))
)


@dataclass(frozen=True)
class Resource:
    path: os.PathLike | str
    remote: bool = False
    resource_dir: os.PathLike | str = DEFAULT_RESOURCE_DIR
    base_url: str = DEFAULT_BASE_URL

    def get(self):
        path = Path(self.path)
        resource_dir = Path(self.resource_dir)
        if self.remote:
            if path.is_absolute():
                path = path.relative_to("/")
            main_resource = Path(path.parts[0])
            relative_resource = path.relative_to(main_resource)
            resource_dir.mkdir(parents=True, exist_ok=True)

            custom_locations = [
                p / main_resource
                for p in custom_remote_search_locations
                if (p / main_resource).exists()
            ]
            if len(custom_locations) > 0:
                return (custom_locations[0] / relative_resource).resolve()

            try:
                resource_path = self._download_resource(
                    main_resource, resource_dir, filelock_timeout=0
                )
            except filelock.Timeout:
                logger.info(
                    "Resource is currently being updated. Waiting for update to finish..."
                )
                resource_path = self._download_resource(main_resource, resource_dir)
            return (resource_path / relative_resource).resolve()
        else:
            return path.resolve()

    @staticmethod
    def parse_resource_spec(
        resource_spec: os.PathLike | str,
        resource_dir: os.PathLike | str = DEFAULT_RESOURCE_DIR,
        base_url: str = DEFAULT_BASE_URL,
    ) -> "Resource":
        """
        If resource_spec has the form "remote:<path>", then a remote resource with the specified path is created.
        Otherwise, resource_spec is interpreted as a local path and a local resource is created.
        :param resource_spec: Resource specification: either a path or a resource identifier, starting with "remote:".
        :param resource_dir:  Directory where resources are stored.
        :param base_url:      Base URL for remote resources.
        :return:
        """
        remote_prefix = "remote:"
        if isinstance(resource_spec, str) and resource_spec.startswith(remote_prefix):
            return Resource(
                resource_spec[len(remote_prefix) :],
                remote=True,
                resource_dir=resource_dir,
                base_url=base_url,
            )
        else:
            return Resource(
                resource_spec,
                remote=False,
                resource_dir=resource_dir,
                base_url=base_url,
            )

    def _download_resource(
        self, path: Path, resource_dir: Path, filelock_timeout: int = -1
    ) -> Path:
        with filelock.FileLock(resource_dir / "update.lock", timeout=filelock_timeout):
            tmp_file = resource_dir / "tmp.zip"
            tmp_file.unlink(missing_ok=True)
            target_path = resource_dir / path
            if not target_path.exists():
                params = urllib.parse.urlencode({"path": str(path)})
                download_url = f"{self.base_url}/download?{params}"

                try:
                    with urllib.request.urlopen(download_url) as response:
                        with tmp_file.open("wb") as f:
                            chunk_size = 4096
                            bar = tqdm.tqdm(
                                desc=f"Downloading {path}", unit="B", unit_scale=True
                            )
                            while (chunk := response.read(chunk_size)) != b"":
                                f.write(chunk)
                                bar.update(len(chunk))
                            bar.close()
                    logger.info(f"Extracting {path}... ")
                    with ZipFile(tmp_file) as zipfile:
                        zipfile.extractall(target_path.parent)
                    logger.info("Done.")
                finally:
                    if tmp_file.exists():
                        tmp_file.unlink()
            return target_path


def get_remote_resource(
    path: os.PathLike | str,
    resource_dir: os.PathLike | str = DEFAULT_RESOURCE_DIR,
    base_url: str = DEFAULT_BASE_URL,
) -> Path:
    return Resource(
        path, remote=True, resource_dir=resource_dir, base_url=base_url
    ).get()


def get_resource(
    resource_spec: os.PathLike | str,
    resource_dir: os.PathLike | str = DEFAULT_RESOURCE_DIR,
    base_url: str = DEFAULT_BASE_URL,
) -> Path:
    return Resource.parse_resource_spec(
        resource_spec, resource_dir=resource_dir, base_url=base_url
    ).get()
