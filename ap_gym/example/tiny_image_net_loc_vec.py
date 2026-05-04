#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

import ap_gym

env = ap_gym.make_vec("TinyImageNetLoc-v0", num_envs=4, render_mode="rgb_array")

obs, _ = env.reset(seed=0)
img = env.render()

fig, axes = plt.subplots(3, env.num_envs, squeeze=False)
glimpse_plot = [
    ax[0].imshow(
        np.zeros(env.observation_space["glimpse"].shape[1:]), vmin=0.0, vmax=1.0
    )
    for ax in axes.T
]
target_glimpse_plot = [
    ax[1].imshow(
        np.zeros(env.observation_space["target_glimpse"].shape[1:]), vmin=0.0, vmax=1.0
    )
    for ax in axes.T
]
render_plot = [ax[2].imshow(np.zeros_like(im)) for ax, im in zip(axes.T, img)]
plt.show(block=False)

for _ in range(1000):
    action = {
        "action": env.inner_action_space.sample(),
        "prediction": env.prediction_space.sample(),
    }

    obs, _, _, _, info = env.step(action)
    print(
        f"Current loss: {env.loss_fn.numpy(action['prediction'], info['prediction']['target'])}"
    )
    for gp, tgp, rp, g, tg, img in zip(
        glimpse_plot,
        target_glimpse_plot,
        render_plot,
        obs["glimpse"],
        obs["target_glimpse"],
        env.render(),
    ):
        gp.set_data(g)
        tgp.set_data(tg)
        rp.set_data(img)
    plt.pause(1 / env.metadata["render_fps"])

env.close()
