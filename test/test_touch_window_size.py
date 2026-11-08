"""All snap and real-snap environments choose every touch from a window of 16 touches."""

import gymnasium as gym

import tactile_mnist  # noqa: F401
from tactile_mnist import TactileRealSnapConfig


def test_snap_variants_sample_16_touch_positions():
    snap_ids = [k for k in gym.registry if "Snap" in k and "RealSnap" not in k]
    assert snap_ids
    for env_id in snap_ids:
        default_config = gym.registry[env_id].kwargs["default_config"]
        assert (
            default_config["snap_touch_positions"]
            == TactileRealSnapConfig.touch_window_size
            == 16
        ), env_id


def test_real_snap_window_matches_the_snap_variants():
    assert TactileRealSnapConfig.touch_window_size == 16
    for env_id in [k for k in gym.registry if "RealSnap" in k]:
        assert "touch_window_size" not in gym.registry[env_id].kwargs["default_config"]
