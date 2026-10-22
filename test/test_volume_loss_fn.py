"""Tests for the normalized Huber loss of the volume estimation environments
(tactile_mnist.tactile_volume_estimation_env.mk_volume_loss_fn)."""

import numpy as np
import pytest
from datasets import load_dataset

from tactile_mnist import SimpleMeshDataset
from tactile_mnist.registration import mk_volume_statistics_dataset
from tactile_mnist.tactile_volume_estimation_env import (
    DEFAULT_HUBER_DELTA,
    get_volume_statistics,
    mk_volume_loss_fn,
)

NUM_OBJECTS = 64


@pytest.fixture(scope="module")
def dataset():
    return SimpleMeshDataset(
        load_dataset("TimSchneider42/tactile-mnist-mnist3d", split="train").select(
            range(NUM_OBJECTS)
        )
    )


@pytest.fixture(scope="module")
def standardized_volumes(dataset):
    statistics = get_volume_statistics(dataset)
    volumes = np.array([dataset[i].mesh.volume for i in range(len(dataset))])
    return statistics, ((volumes - statistics["mean"]) / statistics["std"])[:, None]


@pytest.mark.parametrize("delta", [0.5, DEFAULT_HUBER_DELTA, 2.0])
def test_best_blind_guess_scores_one(dataset, standardized_volumes, delta):
    statistics, targets = standardized_volumes
    loss_fn = mk_volume_loss_fn(dataset, statistics, delta)
    assert loss_fn.blind_guessing_expected_value == pytest.approx(1.0)
    # No constant guess does better than the blind guess, the mean guess included
    for guess in np.linspace(-1.0, 1.0, 41):
        losses = loss_fn.numpy(np.full_like(targets, guess), targets, (len(targets),))
        assert np.mean(losses) >= 1.0 - 1e-9
    assert loss_fn.numpy(targets, targets, (len(targets),)) == pytest.approx(0.0)


def test_evaluation_splits_use_training_statistics():
    assert mk_volume_statistics_dataset("mnist3d", "train") is None
    statistics_dataset = mk_volume_statistics_dataset("mnist3d", "test")
    train_statistics = get_volume_statistics(
        SimpleMeshDataset(
            load_dataset("TimSchneider42/tactile-mnist-mnist3d", split="train")
        )
    )
    assert get_volume_statistics(statistics_dataset) == train_statistics
