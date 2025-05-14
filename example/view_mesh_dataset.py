#!/usr/bin/env python3

import argparse

from datasets import load_dataset

from tactile_mnist import MeshDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="TimSchneider42/tactile-mnist-mnist3d",
        help="Name or path of the dataset to load.",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=int,
        nargs="+",
        choices=range(10),
        help="Label to show examples for.",
    )
    parser.add_argument(
        "-s",
        "--split",
        type=str,
        default="train",
        help="Split of the dataset to load.",
    )
    args = parser.parse_args()

    dataset = MeshDataset(load_dataset(args.dataset, split=args.split))

    if args.label is not None:
        dataset = dataset.filter_labels(args.label)

    for data_point in dataset:
        print(f"Datapoint {data_point.id}")
        data_point.mesh.show()  # data_point.mesh is a trimesh.Trimesh object
