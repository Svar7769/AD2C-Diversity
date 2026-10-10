"""Reusable BenchMARL experiment runner with optional ESC control."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import yaml
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import benchmarl.models
from benchmarl.algorithms import IppoConfig, IsacConfig, MappoConfig, MasacConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.hydra_config import (
    load_algorithm_config_from_hydra,
    load_experiment_config_from_hydra,
    load_model_config_from_hydra,
    load_task_config_from_hydra,
)

from het_control.callbacks.callback import (
    ActionSpaceLoss,
    NormLoggerCallback,
    SndCallback,
    TagCurriculum,
)
from het_control.callbacks.esc_callback import ESCCallback
from het_control.environments.vmas import render_callback
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpiricalConfig


def setup(task_name: str) -> None:
    """Register the custom model and existing task rendering behavior."""
    benchmarl.models.model_config_registry.update(
        {"hetcontrolmlpempirical": HetControlMlpEmpiricalConfig}
    )


def get_experiment(
    cfg: DictConfig,
    esc_config: Optional[Dict[str, Any]] = None,
    visualize_snd: bool = False,
) -> Experiment:
    """Build the experiment; SND plotting is optional."""
    choices = HydraConfig.get().runtime.choices
    task_name = choices.task
    setup(task_name)
    print(f"Algorithm: {choices.algorithm}, Task: {task_name}")
    print(OmegaConf.to_yaml(cfg))

    algorithm_config = load_algorithm_config_from_hydra(cfg.algorithm)
    experiment_config = load_experiment_config_from_hydra(cfg.experiment)
    task_config = load_task_config_from_hydra(cfg.task, task_name)

    if task_name in {
        "vmas/balance", "vmas/ball_passage", "vmas/ball_trajectory",
        "vmas/buzz_wire", "vmas/discovery", "vmas/dispersion",
        "vmas/football", "vmas/navigation", "vmas/reverse_transport",
        "vmas/sampling", "vmas/tag",
    }:
        type(task_config).render_callback = staticmethod(render_callback)

    critic_model_config = load_model_config_from_hydra(cfg.critic_model)
    model_config = load_model_config_from_hydra(cfg.model)

    if isinstance(algorithm_config, (MappoConfig, IppoConfig, MasacConfig, IsacConfig)):
        model_config.probabilistic = True
        model_config.scale_mapping = algorithm_config.scale_mapping
        algorithm_config.scale_mapping = "relu"
    else:
        model_config.probabilistic = False

    callbacks = [SndCallback(), NormLoggerCallback()]
    if esc_config is not None:
        if hasattr(model_config, "desired_snd"):
            model_config.desired_snd = esc_config.get("initial_snd", 0.0)
        callbacks.append(
            ESCCallback(
                control_group=esc_config.get("control_group", "agents"),
                initial_snd=esc_config.get("initial_snd", 0.0),
                dither_magnitude=esc_config.get("dither_magnitude", 0.1),
                dither_frequency=esc_config.get("dither_frequency", 0.5),
                integrator_gain=esc_config.get("integrator_gain", -0.01),
                high_pass_cutoff=esc_config.get("high_pass_cutoff", 0.1),
                low_pass_cutoff=esc_config.get("low_pass_cutoff", 0.05),
                use_adaptive_gain=esc_config.get("use_adaptive_gain", True),
                grad_threshold=esc_config.get("gradient_threshold", 5.0),
                high_gain=esc_config.get("high_gain", -0.015),
                min_snd=esc_config.get("min_snd", 0.0),
                max_snd=esc_config.get("max_snd", 3.0),
                reward_scale=esc_config.get("reward_scale", 1.0),
            )
        )


    action_config = esc_config if esc_config is not None else cfg
    callbacks.append(
        ActionSpaceLoss(
            use_action_loss=action_config.get("use_action_loss", False),
            action_loss_lr=action_config.get("action_loss_lr", 0.001),
        )
    )

    if visualize_snd:
        from het_control.callbacks.sndVisualCallback import SNDVisualizerCallback

        callbacks.append(SNDVisualizerCallback())

    if task_name == "vmas/simple_tag":
        curriculum_config = esc_config if esc_config else cfg
        callbacks.append(
            TagCurriculum(
                curriculum_config.get("simple_tag_freeze_policy_after_frames", 1_000_000),
                curriculum_config.get("simple_tag_freeze_policy", False),
            )
        )

    return Experiment(
        task=task_config,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=cfg.seed,
        config=experiment_config,
        callbacks=callbacks,
    )


def load_esc_config(config_path: str) -> Dict[str, Any]:
    """Read the ESC controller section from its YAML file."""
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle) or {}
    return config.get("esc_controller", {})


def _override_value(value: Any) -> str:
    """Preserve CLI Hydra expressions and encode scalar/list values."""
    return value if isinstance(value, str) else json.dumps(value)


def run_experiment(
    config_path: str,
    config_name: str,
    save_path: str,
    max_frames: int,
    checkpoint_interval: int,
    desired_snd: float = 0.0,
    task_overrides: Optional[Dict[str, Any]] = None,
    esc_config_path: Optional[str] = None,
    use_esc: bool = True,
    seed: Optional[int] = None,
    model_overrides: Optional[Dict[str, Any]] = None,
    experiment_overrides: Optional[Dict[str, Any]] = None,
    visualize_snd: bool = False,
) -> None:
    """Run training with task, model and experiment overrides."""
    esc_config = (
        load_esc_config(esc_config_path)
        if use_esc and esc_config_path is not None
        else None
    )
    if esc_config is not None:
        desired_snd = esc_config.get("initial_snd", desired_snd)

    experiment_params = {
        "max_n_frames": max_frames,
        "checkpoint_interval": checkpoint_interval,
        "save_folder": save_path,
    }
    experiment_params.update(experiment_overrides or {})

    model_params = dict(model_overrides or {})
    model_params["desired_snd"] = desired_snd

    overrides = []
    for category, parameters in (
        ("experiment", experiment_params),
        ("model", model_params),
        ("task", task_overrides or {}),
    ):
        for name, value in parameters.items():
            overrides.append(f"{category}.{name}={_override_value(value)}")

    if seed is not None:
        overrides.append(f"seed={seed}")

    if esc_config is not None:
        for name, default in (
            ("use_action_loss", False),
            ("action_loss_lr", 0.001),
        ):
            value = esc_config.get(name, default)
            overrides.append(f"{name}={_override_value(value)}")

    folder = experiment_params["save_folder"]
    if folder is not None:
        Path(folder).mkdir(parents=True, exist_ok=True)

    GlobalHydra.instance().clear()

    @hydra.main(
        version_base=None,
        config_path=config_path,
        config_name=config_name,
    )
    def hydra_experiment(cfg: DictConfig) -> None:
        experiment = get_experiment(
            cfg, esc_config, visualize_snd=visualize_snd
        )
        experiment.run()

    original_argv = sys.argv
    try:
        sys.argv = ["run_experiment"] + overrides
        hydra_experiment()
        print("Experiment completed successfully.")
    finally:
        sys.argv = original_argv
        GlobalHydra.instance().clear()
