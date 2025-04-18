#!/usr/bin/env python3

import logging

from tactile_mnist import TouchDatasetRoundIterator, TouchDataset, get_resource

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

datasets = TouchDataset.open_all(
    get_resource("remote:tactile_mnist-syn-single-t32-320x240-v0/train")
)
for round_data in TouchDatasetRoundIterator(
    datasets, dataset_prefetch_count=1, shuffle=True
):
    # round_data is a touch dataset
    # Iterate over the touches in the round
    for touch in round_data:
        # Do something with the touch
        print(touch.metadata.unique_id)
