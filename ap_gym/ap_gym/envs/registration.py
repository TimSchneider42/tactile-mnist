from __future__ import annotations

from functools import partial, reduce
from typing import Any, Sequence, Callable, overload

import gymnasium as gym
import gymnasium.utils.passive_env_checker
import numpy as np
from gymnasium.envs.registration import (
    WrapperSpec,
    EnvCreator,
    VectorEnvCreator,
    parse_env_id,
    get_env_id,
    load_env_creator,
)

from ap_gym import (
    BaseActivePerceptionEnv,
    ActivePerceptionWrapper,
    ensure_active_perception_env,
    BaseActivePerceptionVectorEnv,
    ensure_active_perception_vector_env,
    ActiveRegressionLogWrapper,
    ActiveRegressionVectorLogWrapper,
    ActiveClassificationLogWrapper,
    ActiveClassificationVectorLogWrapper,
    idoc,
    VectorToSingleWrapper,
    SparsifyWrapper,
)
from ap_gym.envs.image import DoubleCircleSquareDataset
from .circle_square_catch_or_flee import CircleSquareHideAndSeekVectorWrapper
from .floor_map import FloorMapDatasetRooms, FloorMapDatasetMaze, FloorMapDataset
from .image import (
    HuggingfaceImageClassificationDataset,
    CircleSquareDataset,
    ImageClassificationDataset,
    ImagePerceptionConfig,
)
from .image_classification import ImageClassificationEnv, ImageClassificationVectorEnv
from .image_localization import ImageLocalizationEnv, ImageLocalizationVectorEnv
from .lidar_localization2d import LIDARLocalization2DEnv
from .light_dark import LightDarkEnv
from ap_gym import SparsifyVectorWrapper

ACTIVE_REGRESSION_LOGGER_WRAPPER_SPEC = WrapperSpec(
    "ActiveRegressionLogWrapper", "ap_gym:ActiveRegressionLogWrapper", kwargs={}
)


@overload
def _wrap_env(
    entry_point: EnvCreator | str,
    *wrappers: Callable[[gym.Env], gym.Env],
) -> EnvCreator: ...


@overload
def _wrap_env(
    entry_point: VectorEnvCreator | str,
    *wrappers: Callable[[gym.vector.VectorEnv], gym.vector.VectorEnv],
) -> VectorEnvCreator: ...


def _wrap_env(
    entry_point: EnvCreator | VectorEnvCreator | str | None = None,
    *wrappers: Callable[
        [gym.Env | gym.vector.VectorEnv], gym.Env | gym.vector.VectorEnv
    ],
) -> EnvCreator | VectorEnvCreator | None:
    if entry_point is None:
        return None

    def creator(**kwargs) -> gym.Env:
        if isinstance(entry_point, str):
            env = gym.make(entry_point, **kwargs)
        else:
            env = entry_point(**kwargs)
        for wrapper in wrappers:
            env = wrapper(env)
        return env

    return creator


def register(
    id: str,
    entry_point: EnvCreator | str | None = None,
    reward_threshold: float | None = None,
    nondeterministic: bool = False,
    max_episode_steps: int | None = None,
    order_enforce: bool = True,
    disable_env_checker: bool = False,
    additional_wrappers: tuple[WrapperSpec, ...] = (),
    vector_entry_point: VectorEnvCreator | str | None = None,
    kwargs: dict | None = None,
    register_sparse_version: bool = True,
):
    register_kwargs = dict(
        reward_threshold=reward_threshold,
        nondeterministic=nondeterministic,
        max_episode_steps=max_episode_steps,
        order_enforce=order_enforce,
        disable_env_checker=disable_env_checker,
        kwargs=kwargs,
    )
    gym.register(
        id=id,
        entry_point=entry_point,
        vector_entry_point=vector_entry_point,
        additional_wrappers=additional_wrappers,
        **register_kwargs,
    )
    if register_sparse_version:
        ns, name, version = parse_env_id(id)
        name = f"{name}-sparse"
        full_env_id = get_env_id(ns, name, version)
        if entry_point is not None:

            def new_entry_point(*args, **kwargs):
                env = entry_point(*args, **kwargs)

                # We have to apply the wrappers here to avoid "gymnasium.error.Error: Cannot use `vector_entry_point`
                # vectorization mode with the additional_wrappers parameter in spec being not empty"

                for wrapper_spec in additional_wrappers:
                    env = load_env_creator(wrapper_spec.entry_point)(
                        env=env, **wrapper_spec.kwargs
                    )

                return SparsifyWrapper(env)

        else:
            new_entry_point = entry_point
        gym.register(
            id=full_env_id,
            entry_point=new_entry_point,
            vector_entry_point=_wrap_env(vector_entry_point, SparsifyVectorWrapper),
            additional_wrappers=(),
            **register_kwargs,
        )


def register_image_env(
    name: str,
    entry_point: Callable[[...], gym.Env],
    vector_entry_point: Callable[[...], gym.vector.VectorEnv],
    dataset: ImageClassificationDataset,
    step_limit: int,
    kwargs: dict[str, Any] | None = None,
    single_wrappers: tuple[Callable[[gym.Env], gym.Env]] = (),
    vector_wrappers: tuple[Callable[[gym.vector.VectorEnv], gym.vector.VectorEnv]] = (),
    idoc_fn: Callable[[gym.Env], dict[str, str]] | None = None,
):
    if kwargs is None:
        kwargs = {}
    if idoc_fn is not None:

        def _entry_point(*args, **kwargs):
            env = entry_point(*args, **kwargs)
            return idoc(env, idoc_fn(env))

    else:
        _entry_point = entry_point
    register(
        id=name,
        kwargs=dict(
            image_perception_config=ImagePerceptionConfig(
                dataset=dataset, step_limit=step_limit, **kwargs
            )
        ),
        entry_point=lambda *args, **kwargs: reduce(
            lambda env, wrapper_fn: wrapper_fn(env),
            single_wrappers,
            _entry_point(*args, **kwargs),
        ),
        vector_entry_point=lambda *args, **kwargs: reduce(
            lambda env, wrapper_fn: wrapper_fn(env),
            vector_wrappers,
            vector_entry_point(*args, **kwargs),
        ),
    )


register_image_classification_env = partial(
    register_image_env,
    entry_point=ImageClassificationEnv,
    vector_entry_point=ImageClassificationVectorEnv,
    single_wrappers=(ActiveClassificationLogWrapper,),
    vector_wrappers=(ActiveClassificationVectorLogWrapper,),
)


def register_img_envs(
    fn: Callable,
    name: str,
    datasets: Callable[[str], ImageClassificationDataset],
    step_limit: int,
    kwargs: dict[str, Any] | None = None,
    idoc_fn: Callable[[gym.Env], dict[str, str]] | None = None,
):
    for split in ["train", "test"]:
        split_names = [f"-{split}"]
        if split == "train":
            split_names.append("")
        for split_name in split_names:
            if split_name == "":
                _idoc_fn = idoc_fn
            elif split_name == "train":
                _idoc_fn = lambda env: {
                    **(idoc_fn(env) if idoc_fn is not None else {}),
                    **{"description": f"Alias for {name}-v0."},
                }
            else:
                _idoc_fn = lambda env, _split=split: {
                    **(idoc_fn(env) if idoc_fn is not None else {}),
                    **{
                        "description": f"Uses the {_split} split of {name} instead of the train split."
                    },
                }
            fn(
                name=f"{name}{split_name}-v0",
                dataset=datasets(split),
                step_limit=step_limit,
                kwargs=kwargs,
                idoc_fn=_idoc_fn,
            )


register_image_localization_env = partial(
    register_image_env,
    entry_point=ImageLocalizationEnv,
    vector_entry_point=ImageLocalizationVectorEnv,
    single_wrappers=(ActiveRegressionLogWrapper,),
    vector_wrappers=(ActiveRegressionVectorLogWrapper,),
)


def mk_img_idoc_fn(description: str, image_description: str):
    def idoc_fn(env):
        output = {}
        if description is not None:
            output["description"] = description
        glimpse = env.observation_space["glimpse"]
        dataset = env.config.dataset
        sample = dataset[0][0]
        output["properties"] = {
            "Image type": ("RGB" if glimpse.shape[-1] == 3 else "Grayscale"),
            "# data points": str(len(dataset)),
            "Image size": f"{sample.shape[1]}x{sample.shape[0]}",
            "Glimpse size": f"{glimpse.shape[1]}x{glimpse.shape[0]}",
            "Step limit": str(env.config.step_limit),
            "Image description": image_description,
        }
        return output

    return idoc_fn


def mk_img_class_idoc_fn(description: str, image_description: str):
    _idoc_fn = mk_img_idoc_fn(description, image_description)

    def idoc_fn(env):
        output = _idoc_fn(env)
        output["properties"]["# classes"] = str(env.config.dataset.num_classes)
        desc = output["properties"]["Image description"]
        del output["properties"]["Image description"]
        output["properties"]["Image description"] = desc  # Move to end
        return output

    return idoc_fn


def register_image_classification_envs(
    name: str,
    datasets: Callable[[str], ImageClassificationDataset],
    step_limit: int,
    description: str,
    image_description: str,
    kwargs: dict[str, Any] | None = None,
):
    return register_img_envs(
        register_image_classification_env,
        name,
        datasets,
        step_limit,
        kwargs,
        mk_img_class_idoc_fn(description, image_description),
    )


def register_image_localization_envs(
    name: str,
    datasets: Callable[[str], ImageClassificationDataset],
    step_limit: int,
    description: str,
    image_description: str,
    kwargs: dict[str, Any] | None = None,
):
    return register_img_envs(
        register_image_localization_env,
        name,
        datasets,
        step_limit,
        kwargs,
        mk_img_idoc_fn(description, image_description),
    )


def mk_time_limit(step_limit: int, issue_termination=True) -> WrapperSpec:
    return WrapperSpec(
        "TimeLimit",
        "ap_gym:TimeLimit",
        kwargs=dict(max_episode_steps=step_limit, issue_termination=issue_termination),
    )


def register_lidar_localization_env(
    name: str,
    dataset: FloorMapDataset,
    description: str,
    map_type: str,
    static_map: bool = False,
    step_limit: int = 100,
):
    def mk_env(*args, **kwargs):
        env = LIDARLocalization2DEnv(*args, **kwargs)
        if static_map:
            short_desc = f"{map_type.capitalize()} with static map."
        else:
            short_desc = (
                f"Dynamic {map_type} environment with different maps per episode."
            )
        return idoc(
            env,
            {
                "description": description,
                "properties": {
                    "Map type": map_type.capitalize(),
                    "Static/dynamic": "Static" if static_map else "Dynamic",
                    "Map size": f"{env.dataset.map_width}x{env.dataset.map_height}",
                    "Map description": short_desc,
                },
            },
        )

    register(
        id=name,
        entry_point=mk_env,
        kwargs=dict(dataset=dataset, static_map=static_map),
        additional_wrappers=(
            mk_time_limit(step_limit),
            ACTIVE_REGRESSION_LOGGER_WRAPPER_SPEC,
        ),
    )


def register_circle_square(
    size: int, show_gradient: bool, suffix: str, description: str, step_limit: int = 16
):
    defaul_img_desc = "An image containing either a circle or square."
    register_image_classification_env(
        name=f"CircleSquare{suffix}-v0",
        dataset=CircleSquareDataset(
            image_shape=(size, size), show_gradient=show_gradient
        ),
        step_limit=step_limit,
        idoc_fn=mk_img_class_idoc_fn(
            description.format(env="CircleSquare", extra_info=""),
            defaul_img_desc,
        ),
    )
    register_image_classification_env(
        name=f"CircleSquareInverted{suffix}-v0",
        dataset=CircleSquareDataset(
            image_shape=(size, size),
            show_gradient=show_gradient,
        ),
        step_limit=step_limit,
        idoc_fn=mk_img_class_idoc_fn(
            description.format(
                env="CircleSquareInverted",
                extra_info="In this variant of CircleSquare, the every episode, the label is randomly inverted at the "
                "start. The agent gets a binary signal indicating whether the label is inverted at the "
                "start of the episode. This environment tests whether the agent can remember information "
                "over the course of the episode and use it to correctly solve the task.",
            ),
            defaul_img_desc,
        ),
        kwargs=dict(randomly_invert_labels=True),
    )

    register_image_classification_env(
        name=f"DoubleCircleSquare{suffix}-v0",
        dataset=DoubleCircleSquareDataset(
            image_shape=(size, size),
            show_gradient_a=show_gradient,
            show_gradient_b=show_gradient,
        ),
        step_limit=step_limit,
        idoc_fn=mk_img_class_idoc_fn(
            description.format(env="DoubleCircleSquare", extra_info=""),
            "An image containing either two circles, two squares or one each. The task is to determine "
            "whether there are two of the same shape or one of each shape.",
        ),
    )


def register_envs():
    register_circle_square(
        28,
        True,
        "",
        "In the {env} environment, the agent's objective is to determine whether a given image contains a"
        "circle or a square. The agent has limited visibility, represented by a small movable glimpse that captures "
        "partial views of the image. A visual gradient within the image guides the agent towards the object. "
        "{extra_info}",
    )
    register_circle_square(
        28,
        True,
        "-s28",
        "Alias for {env}-v0.",
    )
    register_circle_square(
        20,
        True,
        "-s20",
        "Variant of {env} with a smaller image size of 20 instead of 28.",
    )
    register_circle_square(
        15,
        True,
        "-s15",
        "Variant of {env} with an even smaller image size of 15 instead of 28.",
    )
    register_circle_square(
        28,
        False,
        "-nograd",
        "Variant of {env} with no gradient as visual aid.",
    )
    register_circle_square(
        20,
        False,
        "-s20-nograd",
        "Variant of {env}-nograd with a smaller image size of 20 instead of 28.",
    )
    register_circle_square(
        15,
        False,
        "-s15-nograd",
        "Variant of {env}-nograd with a smaller image size of 15 instead of 28.",
    )
    register_circle_square(
        28,
        True,
        "-t32",
        "Variant of {env} with a higher time limit of 32 steps instead of 16.",
        step_limit=32,
    )
    register_circle_square(
        28,
        True,
        "-t64",
        "Variant of {env} with a higher time limit of 64 steps instead of 16.",
        step_limit=64,
    )

    register_image_env(
        name="CircleSquareHideAndSeek-v0",
        dataset=CircleSquareDataset(image_shape=(28, 28), show_gradient=True),
        step_limit=32,
        idoc_fn=mk_img_class_idoc_fn(
            "Variant of CircleSquare, in which the agent receives an additional reward for staying close to squares and "
            "far from circles. The time limit is 32 steps instead of 16.",
            "An image containing either a circle or square.",
        ),
        entry_point=lambda *args, **kwargs: VectorToSingleWrapper(
            CircleSquareHideAndSeekVectorWrapper(
                ImageClassificationVectorEnv(*args, **kwargs)
            )
        ),
        vector_entry_point=lambda *args, **kwargs: CircleSquareHideAndSeekVectorWrapper(
            ImageClassificationVectorEnv(*args, **kwargs)
        ),
        single_wrappers=(ActiveClassificationLogWrapper,),
        vector_wrappers=(ActiveClassificationVectorLogWrapper,),
    )

    register_image_env(
        name="CircleSquareHideAndSeekNoPrediction-v0",
        dataset=CircleSquareDataset(image_shape=(28, 28), show_gradient=True),
        step_limit=32,
        idoc_fn=mk_img_class_idoc_fn(
            "Variant of CircleSquareHideAndSeek in which the agent does not need to predict the object class.",
            "An image containing either a circle or square.",
        ),
        entry_point=lambda *args, **kwargs: VectorToSingleWrapper(
            CircleSquareHideAndSeekVectorWrapper(
                ImageClassificationVectorEnv(*args, **kwargs, num_envs=1),
                mask_prediction=True,
            )
        ),
        vector_entry_point=lambda *args, **kwargs: CircleSquareHideAndSeekVectorWrapper(
            ImageClassificationVectorEnv(*args, **kwargs), mask_prediction=True
        ),
    )

    image_env_render_kwargs = dict(
        render_unvisited_opacity=0.5,
        render_visited_opacity=0.25,
    )

    register_image_classification_envs(
        name=f"MNIST",
        datasets=lambda split: HuggingfaceImageClassificationDataset(
            "mnist", channels=1, split=split
        ),
        step_limit=16,
        description="In the MNIST environment, the agent's objective is to classify images of handwritten digits "
        "(0-9). The agent has limited visibility, represented by a small movable glimpse that captures partial views "
        "of the image. It must strategically explore different regions of the image to gather enough information for "
        "accurate classification.",
        image_description="Handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).",
    )

    cifar10_dataset_desc = "Natural images from the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)."
    register_image_classification_envs(
        name="CIFAR10",
        datasets=lambda split: HuggingfaceImageClassificationDataset(
            "cifar10", image_feature_name="img", split=split
        ),
        step_limit=16,
        kwargs=image_env_render_kwargs,
        description="In the CIFAR10 environment, the agent's objective is to classify natural images into 10 classes. "
        "The agent has limited visibility, represented by a small movable glimpse that captures partial views of the "
        "image. It must strategically explore different regions of the image to gather enough information for accurate "
        "classification.",
        image_description=cifar10_dataset_desc,
    )

    CIFAR10_classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    for i in range(2, 11):
        if i == 10:
            desc = "Alias for CIFAR10."
        else:
            desc = f"Variant of CIFAR10, where only the first {i} classes ({', '.join(CIFAR10_classes[:i])}) are used."
        register_image_classification_envs(
            name=f"CIFAR10-c{i}",
            datasets=lambda split: HuggingfaceImageClassificationDataset(
                "cifar10",
                image_feature_name="img",
                split=split,
                filter_labels=CIFAR10_classes[:i],
            ),
            step_limit=16,
            kwargs=image_env_render_kwargs,
            description=desc,
            image_description=cifar10_dataset_desc,
        )

    register_image_classification_envs(
        name=f"TinyImageNet",
        datasets=lambda split: HuggingfaceImageClassificationDataset(
            "zh-plus/tiny-imagenet",
            split=split if split == "train" else "valid",
        ),
        step_limit=16,
        kwargs=dict(sensor_size=(10, 10), **image_env_render_kwargs),
        description="In the TinyImageNet environment, the agent's objective is to classify natural images into 200 "
        "classes. The agent has limited visibility, represented by a small movable glimpse that captures partial views "
        "of the image. It must strategically explore different regions of the image to gather enough information for "
        "accurate classification.\n\nCompared to the CIFAR10 environment, the TinyImageNet dataset contains more "
        "classes and higher resolution images. Also, the glimpse size is larger to account for the higher image "
        "resolution. Consequently, this environment introduces additional complexity compared to CIFAR10.",
        image_description="Natural images from the "
        "[Tiny ImageNet dataset](https://huggingface.co/datasets/zh-plus/tiny-imagenet).",
    )

    register_image_localization_envs(
        name=f"MNISTLoc",
        datasets=lambda split: HuggingfaceImageClassificationDataset(
            "mnist", channels=1, split=split
        ),
        step_limit=16,
        kwargs=image_env_render_kwargs,
        description="In the MNISTLoc environment, the agent's objective is to localize a given glimpse in an MNIST "
        "image. The agent has limited visibility, represented by a small movable glimpse that captures partial views "
        "of the image. It must strategically explore different regions of the image to gather enough information for "
        "accurate localization.",
        image_description="Handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).",
    )

    register_image_localization_envs(
        name=f"CIFAR10Loc",
        datasets=lambda split: HuggingfaceImageClassificationDataset(
            "cifar10", image_feature_name="img", split=split
        ),
        step_limit=16,
        kwargs=image_env_render_kwargs,
        description="In the CIFAR10Loc environment, the agent's objective is to localize a given glimpse in a natural "
        "image. The agent has limited visibility, represented by a small movable glimpse that captures partial views "
        "of the image. It must strategically explore different regions of the image to gather enough information for "
        "accurate localization.",
        image_description="Natural images from the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).",
    )

    register_image_localization_envs(
        name=f"TinyImageNetLoc",
        datasets=lambda split: HuggingfaceImageClassificationDataset(
            "zh-plus/tiny-imagenet",
            split=split if split == "train" else "valid",
        ),
        step_limit=16,
        kwargs=dict(sensor_size=(10, 10), **image_env_render_kwargs),
        description="In the TinyImageNetLoc environment, the agent's objective is to localize a given glimpse in a "
        "natural image. The agent has limited visibility, represented by a small movable glimpse that captures partial "
        "views of the image. It must strategically explore different regions of the image to gather enough information "
        "for accurate classification.\n\nCompared to the CIFAR10Loc environment, the TinyImageNetLoc dataset contains "
        "higher resolution images from more diverse classes. Also, the glimpse size is larger to account for the "
        "higher image resolution. Consequently, this environment introduces additional complexity compared to "
        "CIFAR10Loc.",
        image_description="Natural images from the "
        "[Tiny ImageNet dataset](https://huggingface.co/datasets/zh-plus/tiny-imagenet).",
    )

    register(
        id="LightDark-v0",
        entry_point=LightDarkEnv,
        additional_wrappers=(
            mk_time_limit(50),
            ACTIVE_REGRESSION_LOGGER_WRAPPER_SPEC,
        ),
    )

    register_lidar_localization_env(
        "LIDARLocMazeStatic-v0",
        dataset=FloorMapDatasetMaze(),
        description="In the LIDARLocMazeStatic environment, the agent faces a map with narrow corridors. Hence, it "
        "will always receive information from its LIDAR sensors, but many regions of the maze look alike. The agent "
        "must navigate around the map to gather information and localize itself. In this variant, the map stays "
        "sconstant, meaning that the agent can memorize the layout of the maze over the course of the training.",
        map_type="maze",
        static_map=True,
    )

    register_lidar_localization_env(
        "LIDARLocMaze-v0",
        dataset=FloorMapDatasetMaze(),
        description="In the LIDARLocMaze environment, the agent faces a map with narrow corridors. Hence, it will "
        "always receive information from its LIDAR sensors, but many regions of the maze look alike. The agent must "
        "navigate around the map to gather information and localize itself. In this variant, the maze layout changes "
        "every episode, meaning that the agent has to learn to process the map it is provided as additional input.",
        map_type="maze",
    )

    register_lidar_localization_env(
        "LIDARLocRoomsStatic-v0",
        dataset=FloorMapDatasetRooms(),
        description="In the LIDARLocRooms environment, the agent faces a map with wide open areas. Hence, often it "
        "might not receive any information from its LIDAR sensors if it is in the middle of a large room. The agent "
        "must, thus, navigate around the map to gather information and localize itself. In this variant, the map stays "
        "constant, meaning that the agent can memorize the layout of the rooms over the course of the training.",
        map_type="rooms",
        static_map=True,
    )

    register_lidar_localization_env(
        "LIDARLocRooms-v0",
        dataset=FloorMapDatasetRooms(),
        description="In the LIDARLocRooms environment, the agent faces a map with wide open areas. Hence, often it "
        "might not receive any information from its LIDAR sensors if it is in the middle of a large room. The agent "
        "must, thus, navigate around the map to gather information and localize itself. In this variant, the room "
        "layout changes every episode, meaning that the agent has to learn to process the map it is provided as "
        "additional input.",
        map_type="rooms",
    )


def custom_check_space(
    space: gym.spaces.Space,
    space_type: str,
    check_box_space_fn: Callable[[gym.spaces.Box], None],
):
    """A passive check of the environment action space that should not affect the environment."""
    if not isinstance(space, gym.spaces.Space):
        if str(space.__class__.__base__) == "<class 'gym.spaces.space.Space'>":
            raise TypeError(
                f"Gym is incompatible with Gymnasium, please update the environment {space_type}_space to `{str(space.__class__.__base__).replace('gym', 'gymnasium')}`."
            )
        else:
            raise TypeError(
                f"{space_type} space does not inherit from `gymnasium.spaces.Space`, actual type: {type(space)}"
            )

    elif isinstance(space, gym.spaces.Box):
        check_box_space_fn(space)
    elif isinstance(space, gym.spaces.Discrete):
        assert (
            0 < space.n
        ), f"Discrete {space_type} space's number of elements must be positive, actual number of elements: {space.n}"
        assert (
            space.shape == ()
        ), f"Discrete {space_type} space's shape should be empty, actual shape: {space.shape}"
    elif isinstance(space, gym.spaces.MultiDiscrete):
        assert (
            space.shape == space.nvec.shape
        ), f"Multi-discrete {space_type} space's shape must be equal to the nvec shape, space shape: {space.shape}, nvec shape: {space.nvec.shape}"
        assert np.all(
            0 < space.nvec
        ), f"Multi-discrete {space_type} space's all nvec elements must be greater than 0, actual nvec: {space.nvec}"
    elif isinstance(space, gym.spaces.MultiBinary):
        assert np.all(
            0 < np.asarray(space.shape)
        ), f"Multi-binary {space_type} space's all shape elements must be greater than 0, actual shape: {space.shape}"


def make(
    id: str | gym.envs.registration.EnvSpec,
    max_episode_steps: int | None = None,
    disable_env_checker: bool | None = None,
    **kwargs: Any,
) -> BaseActivePerceptionEnv:
    # We need this to ensure that empty dicts and tuples are allowed
    check_space_orig = gymnasium.utils.passive_env_checker.check_space
    gymnasium.utils.passive_env_checker.check_space = custom_check_space
    try:
        return ensure_active_perception_env(
            gym.make(
                id,
                max_episode_steps=max_episode_steps,
                disable_env_checker=disable_env_checker,
                **kwargs,
            )
        )
    finally:
        gymnasium.utils.passive_env_checker.check_space = check_space_orig


def make_vec(
    id: str | gym.envs.registration.EnvSpec,
    num_envs: int = 1,
    vectorization_mode: gym.VectorizeMode | str | None = None,
    vector_kwargs: dict[str, Any | None] = None,
    wrappers: (
        Sequence[Callable[[BaseActivePerceptionEnv], ActivePerceptionWrapper]] | None
    ) = None,
    **kwargs,
) -> BaseActivePerceptionVectorEnv:
    return ensure_active_perception_vector_env(
        gym.make_vec(
            id, num_envs, vectorization_mode, vector_kwargs, wrappers, **kwargs
        )
    )
