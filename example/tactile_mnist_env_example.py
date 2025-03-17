#!/usr/bin/env python3

import logging

import matplotlib.pyplot as plt
import numpy as np

import ap_gym

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

resolution = (320, 240)

env = ap_gym.make(
    "tactile_mnist:TactileMNIST-train-v0",  # For the test dataset, use "TactileMNIST-test-v0"
    render_mode="rgb_array",
    sensor_output_size=resolution,
)

fig, ax = plt.subplots(1, 2)
img_shape = (resolution[1], resolution[0], 3)
camera_plot = ax[0].imshow(np.zeros(img_shape))
img_plot = ax[1].imshow(np.zeros(img_shape))
plt.show(block=False)

for s in range(100):
    obs, _ = env.reset(seed=s)

    # Generate a circle trajectory
    angles = (
        np.arange(env.spec.max_episode_steps) / env.spec.max_episode_steps * 2 * np.pi
    )
    target_trajectory = np.stack(
        [np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=-1
    )

    for p in target_trajectory:
        action = {
            "action": {"sensor_target_pos_rel": p - obs["sensor_pos"]},
            "prediction": env.prediction_space.sample(),
        }

        obs, _, _, _, info = env.step(action)
        camera_img = env.render()
        img_plot.set_data(obs["sensor_img"])
        camera_plot.set_data(camera_img)
        plt.pause(1 / env.metadata["render_fps"])
