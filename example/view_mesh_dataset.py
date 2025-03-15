import argparse

from tactile_mnist import MeshDataset, get_resource

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="remote:mnist3d_split_v0/train",
        help="Path or resource specification of the mesh dataset.",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=int,
        nargs="+",
        choices=range(10),
        help="Label to show examples for.",
    )
    args = parser.parse_args()

    dataset = MeshDataset.load(get_resource(args.dataset))

    if args.label is not None:
        dataset = dataset.filter_labels(args.label)

    for data_point in dataset:
        print(f"Datapoint {data_point.metadata.id}")
        data_point.mesh.show()  # data_point.mesh is a trimesh.Trimesh object
