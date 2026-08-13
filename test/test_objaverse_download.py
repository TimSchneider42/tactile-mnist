import threading
import time
from pathlib import Path

import numpy as np
import trimesh

from tactile_mnist.objaverse_xl_dataset import load_objaverse_xl_mesh

NUM_THREADS = 4
BOX_EXTENTS = (0.02, 0.03, 0.01)


def test_concurrent_github_downloads_are_serialized(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    obj_content = (
        trimesh.creation.box(extents=BOX_EXTENTS).export(file_type="obj").encode()
    )
    request_urls = []
    request_lock = threading.Lock()

    class FakeResponse:
        content = obj_content

        @staticmethod
        def raise_for_status():
            pass

    def fake_get(url):
        with request_lock:
            request_urls.append(url)
        time.sleep(0.1)
        return FakeResponse()

    monkeypatch.setattr("tactile_mnist.objaverse_xl_dataset.requests.get", fake_get)

    d = {
        "source": "github",
        "fileIdentifier": "https://github.com/user/repo/blob/main/box.obj",
        "sha256": "0" * 64,
    }
    meshes = [None] * NUM_THREADS

    def load(i: int):
        meshes[i] = load_objaverse_xl_mesh(d)

    threads = [threading.Thread(target=load, args=(i,)) for i in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The per-identifier lock must ensure the mesh is only downloaded once
    assert len(request_urls) == 1
    # If any thread had seen a partial file, it would have gotten the fallback mesh instead of the box
    for mesh in meshes:
        np.testing.assert_allclose(sorted(mesh.extents), sorted(BOX_EXTENTS))
    assert not list((tmp_path / ".cache").rglob("*.tmp"))
