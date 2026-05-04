#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

import ap_gym

env = ap_gym.make("CircleSquare-v0", render_mode="rgb_array")

env.reset(seed=0)
img = env.render()

fig, ax = plt.subplots(1, 2)
obs_plot = ax[0].imshow(
    np.zeros(env.observation_space["glimpse"].shape), vmin=0.0, vmax=1.0
)
render_plot = ax[1].imshow(np.zeros_like(img))
plt.show(block=False)

prev_done = False
for _ in range(1000):
    if prev_done:
        obs, _ = env.reset()
        prev_done = False
    else:
        action = {
            "action": env.inner_action_space.sample(),
            "prediction": env.prediction_space.sample(),
        }
        obs, _, terminated, truncated, info = env.step(action)
        prev_done = terminated or truncated
        print(
            f"Current loss: {env.loss_fn.numpy(action['prediction'], info['prediction']['target']):0.2f}"
        )
    obs_plot.set_data(obs["glimpse"])
    render_plot.set_data(env.render())
    plt.pause(1 / env.metadata["render_fps"])

env.close()
