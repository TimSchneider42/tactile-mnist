#!/usr/bin/env python3
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

from tactile_mnist import TouchSingle, TouchSingleDataset, TouchSeqDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="TimSchneider42/tactile-mnist-touch-real-seq-t256-320x240",
        help="Name or path of the dataset to load.",
    )
    parser.add_argument(
        "-s",
        "--split",
        type=str,
        default="train",
        help="Split of the dataset to load.",
    )
    parser.add_argument(
        "-c",
        "--contact-threshold",
        type=float,
        default=0.006,
        help="This value defines a threshold for the height of the gel on the platform. If the gel was lower than this "
        "value, we assume that no contact with the object happened and skip the respective data point.",
    )
    args = parser.parse_args()

    hf_dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    if "sensor_image" in hf_dataset.column_names:
        dataset = TouchSingleDataset(hf_dataset)
    else:
        dataset = TouchSeqDataset(hf_dataset)
    img_plot = None
    for data_point in dataset:
        print(f"Round {data_point.id} on object {data_point.object_id}")
        if isinstance(data_point, TouchSingle):
            if img_plot is None:
                img_plot = plt.imshow(np.zeros_like(data_point.sensor_image[0]))
            num_touches = len(data_point.gel_pose_cell_frame)
        else:
            if img_plot is None:
                video_reader = data_point.sensor_video[0]
                img_plot = plt.imshow(
                    np.zeros_like(next(video_reader)["data"].permute(1, 2, 0).numpy())
                )
                video_reader.seek(0)
            num_touches = len(data_point.gel_pose_cell_frame_seq)
        for i in range(num_touches):
            if isinstance(data_point, TouchSingle):
                touch_height = data_point.gel_pose_cell_frame[i].translation[2]
            else:
                touch_height = min(
                    t.translation[2] for t in data_point.gel_pose_cell_frame_seq[i]
                )
            if touch_height < args.contact_threshold:
                print(
                    f"  Skipping touch {i + 1: 3d}/{num_touches} because of no contact."
                )
                continue
            print(f"  Touch {i + 1: 3d}/{num_touches}")
            if isinstance(data_point, TouchSingle):
                img_plot.set_data(data_point.sensor_image[i])
            else:
                offset = None
                for frame in data_point.sensor_video[i]:
                    if offset is None:
                        offset = time.time() - frame["pts"]
                    img_plot.set_data(frame["data"].permute(1, 2, 0).numpy())
                    plt.pause(max(0.001, frame["pts"] + offset - time.time()))
            plt.pause(1.0)
