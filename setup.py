# Source - https://stackoverflow.com/a/78056725
# Posted by Dev-iL, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-04, License - CC BY-SA 4.0

from pathlib import Path
from setuptools import setup

# This is where you add any fancy path resolution to the local lib:
local_path: str = (Path(__file__).parent / "ap_gym").as_uri()

setup(
    install_requires=[
        f"ap_gym @ {local_path}",
        "filelock",
        "numpy",
        "transformation3d>=1.0.1",
        "tqdm",
        "trimesh[easy]",
        "scikit-robot-pyrender",
        "scipy",
        "requests",
        "opencv-python",
        "av",
        "objaverse"
    ]
)
