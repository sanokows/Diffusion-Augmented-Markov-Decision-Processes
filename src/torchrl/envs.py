import torch
from omegaconf import DictConfig


def make_envs(cfg: DictConfig, device: torch.device, seed: int = None) -> tuple:
    if cfg.env.type == "humanoid_bench":
        from src.env_utils.torch_wrappers.humanoid_bench_env import (
            HumanoidBenchEnv,
        )

        envs = HumanoidBenchEnv(
            cfg.env.name, cfg.hyperparameters.num_envs, device=device
        )
        # Create render environment for video recording
        render_env = HumanoidBenchEnv(
            cfg.env.name, 1, render_mode="rgb_array", device=device
        )
        return envs, envs, render_env
    elif cfg.env.type == "isaaclab":
        from src.env_utils.torch_wrappers.isaaclab_env import IsaacLabEnv

        envs = IsaacLabEnv(
            cfg.env.name,
            device.type,
            cfg.hyperparameters.num_envs,
            cfg=seed,
            action_bounds=cfg.env.action_bounds,
        )
        # For IsaacLab, we don't support separate render env yet
        return envs, envs, envs

    elif cfg.env.type == "mjx":
        from src.env_utils.torch_wrappers.mujoco_playground_env import make_env

        # TODO: Check if re-using same envs for eval could reduce memory usage
        envs, eval_envs = make_env(
            env_name=cfg.env.name,
            seed=seed,
            num_envs=cfg.hyperparameters.num_envs,
            num_eval_envs=cfg.hyperparameters.num_envs,
            device_rank=cfg.platform.device_rank,
            use_domain_randomization=False,
            use_push_randomization=True,
        )
        # For MJX, create a separate render environment  
        render_env, _ = make_env(
            env_name=cfg.env.name,
            seed=seed,
            num_envs=1,
            num_eval_envs=1,
            device_rank=cfg.platform.device_rank,
            use_domain_randomization=False,
            use_push_randomization=False,
        )
        return envs, eval_envs, render_env

    elif cfg.env.type == "maniskill":
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        from mani_skill.utils import gym_utils
        from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
        from src.env_utils.torch_wrappers.maniskill_wrapper import (
            ManiSkillWrapper,
        )

        envs = gym.make(
            cfg.env.name,
            num_envs=cfg.hyperparameters.num_envs,
            reconfiguration_freq=None,
            **cfg.env.env_kwargs,
        )
        eval_envs = gym.make(
            cfg.env.name,
            num_envs=cfg.hyperparameters.num_envs,
            reconfiguration_freq=1,
            **cfg.env.env_kwargs,
        )
        cfg.env.max_episode_steps = gym_utils.find_max_episode_steps_value(envs)
        # heuristic for setting gamma
        cfg.hyperparameters.gamma = 1.0 - 10.0 / cfg.env.max_episode_steps

        if isinstance(envs.action_space, gym.spaces.Dict):
            envs = FlattenActionSpaceWrapper(envs)
            eval_envs = FlattenActionSpaceWrapper(eval_envs)
        envs = ManiSkillVectorEnv(
            envs,
            cfg.hyperparameters.num_envs,
            ignore_terminations=not cfg.env.partial_reset,
            record_metrics=True,
        )
        eval_envs = ManiSkillVectorEnv(
            eval_envs,
            cfg.hyperparameters.num_envs,
            ignore_terminations=True,
            record_metrics=True,
        )
        
        # Create a separate render environment with only 1 env
        render_envs = gym.make(
            cfg.env.name,
            num_envs=1,  # Only 1 environment for rendering
            reconfiguration_freq=1,
            **cfg.env.env_kwargs,
        )
        
        # Add RecordEpisode wrapper if render_dir is specified
        if hasattr(cfg, 'render_dir') and cfg.render_dir is not None:
            from mani_skill.utils.wrappers import RecordEpisode
            from mani_skill.utils import gym_utils
            render_envs = RecordEpisode(
                render_envs, 
                cfg.render_dir, 
                info_on_video=False, 
                save_trajectory=False, 
                max_steps_per_video=gym_utils.find_max_episode_steps_value(render_envs)
            )
        
        if isinstance(render_envs.action_space, gym.spaces.Dict):
            render_envs = FlattenActionSpaceWrapper(render_envs)
        render_envs = ManiSkillVectorEnv(
            render_envs,
            1,  # Only 1 environment
            ignore_terminations=True,
            record_metrics=True,
        )
        
        return ManiSkillWrapper(
            envs,
            max_episode_steps=cfg.env.max_episode_steps,
            partial_reset=cfg.env.partial_reset,
            device=device.type,
        ), ManiSkillWrapper(
            eval_envs,
            max_episode_steps=cfg.env.max_episode_steps,
            partial_reset=cfg.env.partial_reset,
            device=device.type,
        ), ManiSkillWrapper(
            render_envs,  # Use dedicated single render env
            max_episode_steps=cfg.env.max_episode_steps,
            partial_reset=cfg.env.partial_reset,
            device=device.type,
        )
    else:
        raise ValueError(
            f"Unknown environment type: {cfg.env.type}. Supported types are 'humanoid_bench', 'isaaclab', 'maniskill', and 'mjx'."
        )
