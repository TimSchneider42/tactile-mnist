import argparse

import matplotlib.pyplot as plt
import numpy as np

from tactile_mnist import TouchDataset, get_resource, TouchSingle, TouchSeq

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="remote:tactile_mnist_real_seq_t256_v0/train",
        help="Path or resource specification of the mesh dataset.",
    )
    parser.add_argument(
        "-t",
        "--touch-only",
        action="store_true",
        help="Skip images in which the robot was not in contact with the object.",
    )
    args = parser.parse_args()

    with TouchDataset(get_resource(args.dataset)) as dataset:
        data_idx = list(range(len(dataset)))
        if args.touch_only:
            depths = np.concatenate(
                [
                    dataset.metadata[i].gel_positions_cell_frame_seq[:, 2]
                    for i in data_idx
                ]
            )
            min_depth = np.min(depths)
            max_depth = np.max(depths)
            touch_thresh = min_depth + 0.1 * (max_depth - min_depth)
            data_idx = [
                i
                for i in data_idx
                if all(
                    t.translation[2] >= touch_thresh
                    for t in dataset.metadata[i].gel_pose_cell_frame_seq
                )
            ]
            dataset = dataset[data_idx]
        img_plot = plt.imshow(np.zeros_like(dataset[0].sensor_image_seq[0]))
        plt.show(block=False)
        for data_point in dataset:
            print(f"Datapoint {data_point.metadata.id}")
            if isinstance(data_point, TouchSingle):
                img_plot.set_data(data_point.sensor_image)
            else:
                assert isinstance(data_point, TouchSeq)
                for img in data_point.sensor_image_seq:
                    img_plot.set_data(img)
                    plt.pause(1 / 25)
            plt.pause(1.0)
