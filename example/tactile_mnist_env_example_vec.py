import logging

import matplotlib.pyplot as plt
import numpy as np

import ap_gym

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

resolution = (320, 240)

env = ap_gym.make_vec(
    "tactile_mnist:TactileMNIST-train-v0",  # For the test dataset, use "TactileMNIST-test-v0"
    num_envs=4,
    render_mode="rgb_array",
    sensor_output_size=resolution,
)

fig, ax = plt.subplots(2, env.num_envs, squeeze=False)
img_shape = (resolution[1], resolution[0], 3)
camera_plot = [ax[0, i].imshow(np.zeros(img_shape)) for i in range(env.num_envs)]
img_plot = [ax[1, i].imshow(np.zeros(img_shape)) for i in range(env.num_envs)]
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
        for i in range(env.num_envs):
            img_plot[i].set_data(obs["sensor_img"][i])
            camera_plot[i].set_data(camera_img[i])
        plt.pause(1 / env.metadata["render_fps"])
