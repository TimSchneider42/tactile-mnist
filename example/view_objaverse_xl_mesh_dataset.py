#!/usr/bin/env python3

import random

from datasets import load_dataset

from tactile_mnist import ObjaverseXLMeshDataset

if __name__ == "__main__":

    dataset = ObjaverseXLMeshDataset(
        load_dataset("allenai/objaverse-xl", split="train")
    )

    idx = list(range(len(dataset)))
    random.shuffle(idx)
    for i in idx:
        print(f"Datapoint {dataset[i].id}")
        dataset[i].mesh.show()  # data_point.mesh is a trimesh.Trimesh object
