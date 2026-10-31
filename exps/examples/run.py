#!/usr/bin/env python3

import argparse
import os
import sys
from typing import Any

import torch
from habitat_sim.sensor import SensorType

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_PARENT = os.path.abspath(os.path.join(PROJECT_ROOT, "../.."))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_PARENT)

from VisFly.envs.GateHoverEnv import GateHoverEnv
from VisFly.envs.LandingEnv import LandingEnv
from VisFly.envs.MultiNavigationEnv import MultiNavigationEnv
from VisFly.envs.NavigationEnv import NavigationEnv
from VisFly.utils.algorithms.PPO import PPO
from VisFly.utils.algorithms.image_buffer import ImageDictRolloutBuffer
from VisFly.utils.common import load_yaml_config
from VisFly.utils.policies import extractors


ENV_ALIAS = {
    "cluttered_flight": NavigationEnv,
    "crossing": MultiNavigationEnv,
    "landing": LandingEnv,
    "gate_hover_teacher_nominal": GateHoverEnv,
    "gate_hover_teacher": GateHoverEnv,
    "gate_hover_camera": GateHoverEnv,
}

ALG_ALIAS = {
    "PPO": PPO,
}

EXTRACTOR_ALIAS = {
    "StateTargetImageExtractor": extractors.StateTargetImageExtractor,
    "SwarmStateTargetImageExtractor": extractors.SwarmStateTargetImageExtractor,
    "FlexibleExtractor": extractors.FlexibleExtractor,
    "GateImageExtractor": extractors.GateImageExtractor,
}

BUFFER_ALIAS = {
    "ImageDictRolloutBuffer": ImageDictRolloutBuffer,
}

TORCH_ALIAS = {
    "Adam": torch.optim.Adam,
    "ReLU": torch.nn.ReLU,
}

SENSOR_ALIAS = {
    "COLOR": SensorType.COLOR,
    "DEPTH": SensorType.DEPTH,
    "SEMANTIC": SensorType.SEMANTIC,
}


def parse_args(default_env: str = "cluttered_flight"):
    parser = argparse.ArgumentParser(description="Run VisFly examples")
    parser.add_argument("--comment", "-c", type=str, default=None)
    parser.add_argument("--train", "-t", type=int, default=1)
    parser.add_argument("--algorithm", "-a", type=str, default="PPO")
    parser.add_argument("--env", "-e", type=str, default=default_env)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--weight", "-w", type=str, default=None)
    return parser


def resolve_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: resolve_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_config(item) for item in value]
    if isinstance(value, str):
        if value in SENSOR_ALIAS:
            return SENSOR_ALIAS[value]
        if value in EXTRACTOR_ALIAS:
            return EXTRACTOR_ALIAS[value]
        if value in TORCH_ALIAS:
            return TORCH_ALIAS[value]
        if value in BUFFER_ALIAS:
            return BUFFER_ALIAS[value]
    return value


def main(default_env: str = "cluttered_flight") -> None:
    args = parse_args(default_env).parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(base_dir, "saved", args.env)
    os.makedirs(save_folder, exist_ok=True)

    alg_config = load_yaml_config(os.path.join(base_dir, "alg_cfgs", args.env, f"{args.algorithm}.yaml"))
    env_config = load_yaml_config(os.path.join(base_dir, "env_cfgs", f"{args.env}.yaml"))
    alg_config = resolve_config(alg_config)
    env_config = resolve_config(env_config)
    if not args.train:
        env_config["eval_env"]["visual"] = True

    if args.train:
        env = ENV_ALIAS[args.env](**env_config["env"])
        alg_cls = ALG_ALIAS[args.algorithm]
        if args.weight is not None:
            model = alg_cls.load(os.path.join(save_folder, args.weight), env=env)
        else:
            model = alg_cls(
                env=env,
                seed=args.seed,
                comment=args.comment,
                save_path=save_folder,
                **alg_config["algorithm"],
            )
        learn_config = dict(alg_config["learn"])
        callbacks = []
        checkpoint_freq = learn_config.pop("checkpoint_freq", None)  # in environment steps
        if checkpoint_freq:
            from stable_baselines3.common.callbacks import CheckpointCallback
            callbacks.append(CheckpointCallback(
                save_freq=max(1, checkpoint_freq // env.num_envs),
                save_path=model.policy_save_path[:-len(".zip")] + "_checkpoints", name_prefix="ckpt"))
        eval_freq = learn_config.pop("eval_freq", None)  # in environment steps
        if eval_freq:
            # flights of the current policy in a separate env: eval/* in TensorBoard, picture and video
            # in <run>_eval/ (tools/policy_viz.py)
            from tools.policy_viz import EvalCallback
            callbacks.append(EvalCallback(env_config["env"], eval_freq,
                                          out_dir=model.policy_save_path[:-len(".zip")] + "_eval"))
        if callbacks:
            from stable_baselines3.common.callbacks import CallbackList
            learn_config["callback"] = CallbackList(callbacks)
        model.learn(**learn_config)
        model.save()
        env.close()
        return

    if args.weight is None:
        raise ValueError("Testing requires --weight/-w.")

    print("creating eval env...", flush=True)
    eval_env = ENV_ALIAS[args.env](**env_config["eval_env"])
    print("loading model...", flush=True)
    model = ALG_ALIAS[args.algorithm].load(os.path.join(save_folder, args.weight), env=eval_env)
    from VisFly.exps.examples.test.navigation.test import Test as navigation_test

    print("running test...", flush=True)
    test_handle = navigation_test(
        model=model,
        save_path=os.path.join(save_folder, "test"),
        name=args.weight,
    )
    test_handle.test(**alg_config["test"])


if __name__ == "__main__":
    main()
