from tactile_mnist.registration import merge_config
from tactile_mnist.sensor_noise import SensorNoiseConfig


def test_plain_override_replaces_default():
    assert merge_config({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}


def test_sensor_noise_mapping_updates_fields():
    base = SensorNoiseConfig(contact_shift_mean=0.0006, real_base_depth_dataset="x")
    merged = merge_config(
        {"sensor_noise": base}, {"sensor_noise": {"real_base_depth_dataset": None}}
    )
    assert merged["sensor_noise"] == SensorNoiseConfig(
        contact_shift_mean=0.0006, real_base_depth_dataset=None
    )
    assert base.real_base_depth_dataset == "x"


def test_sensor_noise_mapping_without_default_builds_config():
    merged = merge_config(
        {"sensor_noise": None}, {"sensor_noise": {"contact_shift_std": 0.001}}
    )
    assert merged["sensor_noise"] == SensorNoiseConfig(contact_shift_std=0.001)


def test_sensor_noise_instance_replaces_default():
    new = SensorNoiseConfig()
    assert (
        merge_config(
            {"sensor_noise": SensorNoiseConfig(contact_blur_prob=0.3)},
            {"sensor_noise": new},
        )["sensor_noise"]
        is new
    )
