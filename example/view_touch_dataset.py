#!/usr/bin/env python3

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
        default="remote:tactile_mnist-real-seq-t256-320x240-v0/train",
        help="Path or resource specification of the mesh dataset.",
    )
    parser.add_argument(
        "-t",
        "--touch-only",
        action="store_true",
        help="Skip images in which the robot was not in contact with the object.",
    )
    args = parser.parse_args()

    for dataset in TouchDataset.open_all(get_resource(args.dataset)):
        with dataset as dataset_loaded:
            data_idx = list(range(len(dataset_loaded)))
            if args.touch_only:
                min_depths = np.array(
                    [
                        np.min(
                            dataset_loaded.metadata[i].gel_position_cell_frame_seq[:, 2]
                        )
                        for i in data_idx
                    ]
                )
                min_min_depth = np.min(min_depths)
                max_min_depth = np.max(min_depths)
                touch_thresh = min_min_depth + 0.4 * (max_min_depth - min_min_depth)
                data_idx = np.argwhere(min_depths > touch_thresh).flatten()
                dataset_loaded = dataset_loaded[data_idx]
            dp0 = dataset_loaded[0]
            if isinstance(dp0, TouchSingle):
                img_plot = plt.imshow(np.zeros_like(dp0.sensor_image))
            else:
                img_plot = plt.imshow(
                    np.zeros_like(dataset_loaded[0].sensor_image_seq[0])
                )
            plt.show(block=False)
            for data_point in dataset_loaded:
                print(
                    f"Round {data_point.metadata.round_id}, touch {data_point.metadata.touch_no} on object "
                    f"{data_point.metadata.object_id}"
                )
                if isinstance(data_point, TouchSingle):
                    img_plot.set_data(data_point.sensor_image)
                else:
                    assert isinstance(data_point, TouchSeq)
                    for img in data_point.sensor_image_seq:
                        img_plot.set_data(img)
                        plt.pause(1 / 25)
                plt.pause(1.0)
