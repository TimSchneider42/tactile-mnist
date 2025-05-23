#!/usr/bin/env python3

import logging

import matplotlib.pyplot as plt
import numpy as np

import ap_gym

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# For the test dataset, use "TactileMNIST-test-v0", or for the CycleGAN variant use "TactileMNIST-CycleGAN-v0"
env = ap_gym.make("tactile_mnist:TactileMNIST-v0")
obs, _ = env.reset(seed=0)

fig, ax = plt.subplots(1, 2)
camera_plot = ax[0].imshow(env.render())
img_plot = ax[1].imshow(np.zeros(env.observation_space["sensor_img"].shape))
plt.show(block=False)

for _ in range(100):
    # Generate a circle trajectory
    angles = (
        np.arange(env.spec.max_episode_steps) / env.spec.max_episode_steps * 2 * np.pi
    )
    target_trajectory = np.stack(
        [np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=-1
    )

    terminated = False
    for p in target_trajectory:
        action = {
            "action": {"sensor_target_pos_rel": p - obs["sensor_pos"]},
            "prediction": env.prediction_space.sample(),
        }

        obs, _, terminated, _, info = env.step(action)
        camera_img = env.render()
        img_plot.set_data(obs["sensor_img"])
        camera_plot.set_data(camera_img)
        plt.pause(1 / env.metadata["render_fps"])
    assert terminated
    obs, _ = env.reset()
