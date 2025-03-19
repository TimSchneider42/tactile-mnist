#!/usr/bin/env python3

import logging

import matplotlib.pyplot as plt
import numpy as np

import ap_gym

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# For the test dataset, use "tactile_mnist:TactileMNIST-test-v0"
env = ap_gym.make_vec("tactile_mnist:TactileMNIST-v0", num_envs=4)
obs, _ = env.reset(seed=0)

fig, axes = plt.subplots(2, env.num_envs, squeeze=False)
img_plot = [
    ax.imshow(np.zeros(env.single_observation_space["sensor_img"].shape))
    for ax in axes[0]
]
camera_plot = [ax.imshow(img) for img, ax in zip(env.render(), axes[1])]
plt.show(block=False)

terminated = np.zeros(env.num_envs, dtype=bool)
for s in range(100):
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

        obs, _, terminated, _, info = env.step(action)
        camera_img = env.render()
        for i in range(env.num_envs):
            img_plot[i].set_data(obs["sensor_img"][i])
            camera_plot[i].set_data(camera_img[i])
        plt.pause(1 / env.metadata["render_fps"])
    assert np.all(terminated)
    # This will trigger the reset of the environments
    obs, _, terminated, _, _ = env.step(env.action_space.sample())
