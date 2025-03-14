import logging

from tactile_mnist import TouchDatasetRoundIterator, TouchDataset, get_remote_resource

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

datasets = TouchDataset.load_all(
    get_remote_resource("tactile_mnist_syn_single_t32_v0/train")
)
for round_data in TouchDatasetRoundIterator(
    datasets, dataset_prefetch_count=1, shuffle=True
):
    # round_data is a touch dataset
    # Iterate over the touches in the round
    for touch in round_data:
        # Do something with the touch
        print(touch.metadata.unique_id)
