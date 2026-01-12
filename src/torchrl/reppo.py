from dataclasses import dataclass, replace
import functools
import os
import random
import sys
import copy
import time
from pathlib import Path

import numpy as np
import tqdm
from omegaconf import DictConfig, OmegaConf

import wandb

from src.torchrl.reppo_util import EmpiricalNormalization, hl_gauss
from src.torchrl.trajectory_utils import save_and_plot_trajectories

try:
    # Required for avoiding IsaacGym import error
    import isaacgym
except ImportError:
    pass

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from tensordict import TensorDict
from torch.amp import GradScaler
from src.torchrl.envs import make_envs
from src.networks.torch_models import Actor, Critic


torch.set_float32_matmul_precision("high")
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"


@dataclass(slots=True)
class TrainState:
    device: torch.device
    obs: torch.Tensor
    critic_obs: torch.Tensor
    actor: Actor
    old_actor: Actor
    critic: Critic
    normalizer: EmpiricalNormalization
    critic_normalizer: EmpiricalNormalization
    actor_optimizer: optim.Optimizer
    critic_optimizer: optim.Optimizer
    scaler: GradScaler

    def compile(self):
        self.actor.compile()
        self.old_actor.compile()
        self.critic.compile()
        self.normalizer.compile()
        self.critic_normalizer.compile()


def get_autocast_context(cfg: DictConfig):
    amp_enabled = (
        cfg.platform.amp_enabled and cfg.platform.cuda and torch.cuda.is_available()
    )
    amp_device = (
        "cuda"
        if cfg.platform.cuda and torch.cuda.is_available()
        else "mps"
        if cfg.platform.cuda and torch.backends.mps.is_available()
        else "cpu"
    )
    amp_dtype = torch.bfloat16 if cfg.platform.amp_dtype == "bf16" else torch.float32
    return functools.partial(
        torch.amp.autocast,
        device_type=amp_device,
        dtype=amp_dtype,
        enabled=amp_enabled,
    )


def make_collect_fn(cfg: DictConfig, env, env_type: str):
    autocast = get_autocast_context(cfg)
    asymmetric_obs = env.asymmetric_obs

    def collect_fn(
        train_state: TrainState,
    ) -> tuple[TrainState, TensorDict, list[dict]]:
        transitions = []
        info_list = []
        obs = train_state.obs
        critic_obs = train_state.critic_obs

        for _ in range(cfg.hyperparameters.num_steps):
            with autocast():
                norm_obs = train_state.normalizer(obs)
                norm_critic_obs = train_state.critic_normalizer(critic_obs)
                with torch.inference_mode():
                    pi, _, _, _ = train_state.actor(norm_obs)
                    actions = pi.sample()

            # next_obs, rewards, dones, infos = env.step(actions)
            # truncations = infos["time_outs"]

            if env_type == "maniskill":
                next_obs, rewards, dones, truncations, infos = env.step(actions)
            elif env_type == "humanoid_bench":
                next_obs, rewards, dones, infos = env.step(actions)
                truncations = infos["time_outs"]
            else:
                next_obs, rewards, dones, truncations, infos = env.step(actions)

            if asymmetric_obs:
                next_critic_obs = infos["observations"]["critic"]
            else:
                next_critic_obs = next_obs

            with (
                torch.inference_mode(),
                autocast(),
            ):
                if (
                    cfg.env.get("has_final_obs", False)
                    and cfg.env.get("partial_reset", False)
                    and "final_observation" in infos
                ):
                    _next_obs = infos["final_observation"]
                    _next_critic_obs = _next_obs
                else:
                    _next_obs = next_obs
                    _next_critic_obs = next_critic_obs
                norm_next_obs = train_state.normalizer(_next_obs)
                next_pi, _, temperature, _ = train_state.actor(norm_next_obs)
                next_actions = next_pi.sample()
                next_log_probs = next_pi.log_prob(
                    next_actions.clip(-1 + 1e-6, 1 - 1e-6)
                ).sum(-1)
                norm_next_critic_obs = train_state.critic_normalizer(_next_critic_obs)
                next_value, _, _, next_embedding = train_state.critic(
                    norm_next_critic_obs, next_actions
                )
                rewards = (
                    rewards - cfg.hyperparameters.gamma * next_log_probs * temperature
                )

            transitions.append(
                TensorDict(
                    {
                        "observations": norm_obs,
                        "critic_observations": norm_critic_obs,
                        "actions": actions,
                        "log_probs": pi.log_prob(actions.clip(-0.999, 0.999)).sum(-1),
                        "rewards": rewards.unsqueeze(-1),
                        "next_embeddings": next_embedding,
                        "next_values": next_value.unsqueeze(-1),
                        "dones": dones.unsqueeze(-1).float(),
                        "truncations": truncations.unsqueeze(-1).float(),
                    },
                    batch_size=(env.num_envs,),
                )
            )
            info_list.append(infos)
            obs = next_obs
            critic_obs = next_critic_obs

        train_state = replace(train_state, obs=obs, critic_obs=critic_obs)
        return (
            train_state,
            torch.stack(transitions, dim=0),
            info_list,
        )

    return collect_fn


def make_postprocess_fn(cfg: DictConfig, env):
    @torch.compiler.disable()
    def compute_gve(rewards, dones, truncated, next_values, device: torch.device):
        gves = []
        last_gve = 0
        truncated[-1] = 1.0
        for t in reversed(range(cfg.hyperparameters.num_steps)):
            lambda_sum = (
                cfg.hyperparameters.lmbda * last_gve
                + (1.0 - cfg.hyperparameters.lmbda) * next_values[t]
            )
            delta = cfg.hyperparameters.gamma * torch.where(
                truncated[t].bool(), next_values[t], (1.0 - dones[t]) * lambda_sum
            )
            last_gve = rewards[t] + delta
            gves.insert(0, last_gve)
        return gves

    def postprocess(train_state: TrainState, transition: TensorDict):
        gve = compute_gve(
            rewards=transition["rewards"],
            dones=transition["dones"],
            truncated=transition["truncations"],
            next_values=transition["next_values"],
            device=train_state.device,
        )

        # Flatten all time and environment dimensions into a single batch dimension
        data = TensorDict(
            {
                "observations": transition["observations"],
                "critic_observations": transition["critic_observations"],
                "actions": transition["actions"],
                "rewards": transition["rewards"],
                "next_embeddings": transition["next_embeddings"],
                "next_values": transition["next_values"],
                "dones": transition["dones"],
                "truncations": transition["truncations"],
                "gve": torch.stack(gve),
            },
            batch_size=(
                cfg.hyperparameters.num_steps,
                cfg.hyperparameters.num_envs,
            ),
            device=train_state.device,
        )
        return data.float().flatten(0, 1).detach()

    return postprocess


def make_critic_update_fn(cfg: DictConfig, train_state: TrainState):
    autocast = get_autocast_context(cfg)

    def update(data: TensorDict):
        qnet = train_state.critic
        q_optimizer = train_state.critic_optimizer

        with autocast():
            critic_observations = data["critic_observations"]
            actions = data["actions"]
            targets = data["gve"]
            target_embeddings = data["next_embeddings"]
            truncations = data["truncations"].squeeze(-1)
            if cfg.env.get("partial_reset", False):
                truncation_mask = torch.ones_like(
                    truncations, dtype=torch.bool, device=train_state.device
                )
            else:
                truncation_mask = 1.0 - truncations
            qf_target_dist = hl_gauss(
                targets,
                cfg.hyperparameters.vmin,
                cfg.hyperparameters.vmax,
                cfg.hyperparameters.num_bins,
            )

            _, qf1, embedding, _ = qnet(critic_observations, actions)
            qf_loss = -(
                truncation_mask
                * torch.sum(qf_target_dist * F.log_softmax(qf1, dim=-1), dim=-1)
            ).mean()
            embedding_loss = (
                truncation_mask
                * F.mse_loss(
                    embedding,
                    target_embeddings,
                    reduction="none",
                ).mean(dim=-1)
            ).mean()

            qf_loss = qf_loss + cfg.hyperparameters.aux_loss_mult * embedding_loss

        q_optimizer.zero_grad(set_to_none=True)
        train_state.scaler.scale(qf_loss).backward()
        train_state.scaler.unscale_(q_optimizer)

        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            qnet.parameters(), max_norm=cfg.hyperparameters.max_grad_norm
        )
        train_state.scaler.step(q_optimizer)
        train_state.scaler.update()
        logs_dict = {
            "critic_grad_norm": critic_grad_norm.detach(),
            "qf_loss": qf_loss.detach(),
            "qf_max": targets.max().detach(),
            "qf_min": targets.min().detach(),
            "qf_mean": targets.mean().detach(),
            "embedding_loss": embedding_loss.detach(),
        }
        return logs_dict

    return update


def make_actor_update_fn(cfg: DictConfig, train_state: TrainState):
    autocast = get_autocast_context(cfg)

    def update(data: TensorDict):
        actor = train_state.actor
        old_actor = train_state.old_actor
        qnet = train_state.critic
        actor_optimizer = train_state.actor_optimizer
        scaler = train_state.scaler
        critic_obs = data["critic_observations"]
        with autocast():
            pi, _, temperature, beta = actor(data["observations"])
            actions = pi.rsample()
            log_probs = pi.log_prob(actions.clip(-1 + 1e-6, 1 - 1e-6)).sum(-1)
            entropy = -log_probs
            qf, _, _, _ = qnet(critic_obs, actions)
            actor_loss = -qf + temperature.detach() * log_probs

            # compute KL
            old_pi, _, _, _ = old_actor(data["observations"])
            kl_samples = getattr(cfg.hyperparameters, 'kl_action_rep', 16)  # Default to 16 if not specified
            old_pi_actions = old_pi.sample((kl_samples,)).clip(-1 + 1e-6, 1 - 1e-6)
            old_log_probs = old_pi.log_prob(old_pi_actions).sum(-1).mean(0)
            new_pi_log_probs = pi.log_prob(old_pi_actions).sum(-1).mean(0)
            kl = old_log_probs - new_pi_log_probs

            if cfg.hyperparameters.actor_kl_clip_mode == "clipped":
                actor_loss = torch.where(
                    kl < cfg.hyperparameters.kl_bound,
                    actor_loss,
                    kl * beta.detach(),
                ).mean()
            elif cfg.hyperparameters.actor_kl_clip_mode == "full":
                actor_loss = actor_loss + kl * beta.detach()
            elif cfg.hyperparameters.actor_kl_clip_mode == "value":
                actor_loss = actor_loss
            else:
                raise ValueError(
                    f"Unknown actor kl clip mode: {cfg.hyperparameters.actor_kl_clip_mode}"
                )

            # temperature updates
            target_entropy = (
                actions.shape[-1] * cfg.hyperparameters.ent_target_mult
            )  # -0.5 * np.prod(envs.action_space.shape)
            entropy_loss = (target_entropy + entropy).detach().mean() * temperature

            lagrangian_loss = (
                -beta * (kl - cfg.hyperparameters.kl_bound).mean().detach()
            )

            actor_loss = (actor_loss + entropy_loss + lagrangian_loss).mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor.parameters(), max_norm=cfg.hyperparameters.max_grad_norm
        )
        scaler.step(actor_optimizer)
        scaler.update()
        logs_dict = {
            "actor_grad_norm": actor_grad_norm.detach(),
            "actor_loss": actor_loss.detach(),
            "kl": kl.detach(),
            "entropy": entropy.detach(),
            "temperature": temperature.detach(),
            "lagrangian": beta.detach(),
            "entropy_loss": entropy_loss.detach(),
            "lagrangian_loss": lagrangian_loss.detach(),
        }
        return logs_dict

    return update


def make_evaluate_fn(cfg: DictConfig, eval_envs, env_type: str):
    autocast = get_autocast_context(cfg)

    @torch.inference_mode()
    def evaluate(
        train_state: TrainState, stochastic_eval: bool = False
    ) -> tuple[int | float | bool, int | float | bool]:
        train_state.normalizer.eval()
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=train_state.device)
        episode_lengths = torch.zeros(num_eval_envs, device=train_state.device)
        done_masks = torch.zeros(
            num_eval_envs, dtype=torch.bool, device=train_state.device
        )

        if cfg.env.type == "isaaclab" or cfg.env.asymmetric_obs:
            ((obs, *_), _) = eval_envs.reset(random_start_init=False)
        else:
            if cfg.env.type == "maniskill":
                (obs, *_) = eval_envs.reset()
                critic_obs = obs
            else:
                obs = eval_envs.reset()
                critic_obs = obs

        # Run for a fixed number of steps
        for i in range(eval_envs.max_episode_steps):
            with autocast():
                obs = train_state.normalizer(obs)
                action_dist, det_actions, _, _ = train_state.actor(obs)
            if stochastic_eval:
                actions = action_dist.sample()
            else:
                actions = det_actions

            if env_type == "maniskill":
                next_obs, rewards, dones, _, infos = eval_envs.step(actions)
            elif env_type == "humanoid_bench":
                next_obs, rewards, dones, infos = eval_envs.step(actions)
                truncations = infos["time_outs"]
            else:
                next_obs, rewards, dones, _, infos = eval_envs.step(actions)

            episode_returns = torch.where(
                ~done_masks, episode_returns + rewards, episode_returns
            )
            episode_lengths = torch.where(
                ~done_masks, episode_lengths + 1, episode_lengths
            )
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs

        train_state.normalizer.train()

        if cfg.env.type == "maniskill":
            # combine log_infos
            info = {
                "info_return": infos["log_info"]["return"].mean(),
                "episode_len": infos["log_info"]["episode_len"].float().mean(),
                "success": infos["log_info"]["success"].float().mean(),
                "return": episode_returns.mean().item(),
            }
        else:
            info = {}

        return episode_returns.mean().item(), episode_lengths.mean().item(), info

    return evaluate


def make_render_fn(cfg: DictConfig, render_env, env_type: str):
    """Create a render function for video recording and trajectory tracking."""
    autocast = get_autocast_context(cfg)
    
    def render_with_rollout(train_state: TrainState, stochastic_eval: bool = False, track_trajectory: bool = False):
        """Perform a rollout and record video frames. Optionally track end-effector trajectories."""
        train_state.normalizer.eval()
        
        # Initialize trajectory tracking
        trajectories = [] if track_trajectory else None
        
        if env_type == "humanoid_bench":
            obs = render_env.reset()
            renders = [render_env.render()]
        elif env_type == "maniskill":
            # For ManiSkill, use the built-in recording capabilities
            # The render environment should already have RecordEpisode wrapper if configured
            (obs, *_) = render_env.reset()
            renders = []
            
            # Check if environment has recording capability
            has_recording = hasattr(render_env, 'unwrapped') and hasattr(render_env.unwrapped, '_record_episode')
            if has_recording:
                print("Using ManiSkill built-in recording")
            
        elif env_type in ["isaaclab", "mtbench"]:
            # For these environments, we don't support separate rendering yet
            train_state.normalizer.train()
            return []
        elif env_type == "mjx":
            obs = render_env.reset()
            try:
                import jax.numpy as jnp
                render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            except (ImportError, AttributeError):
                pass
            renders = [render_env.state]
        else:
            obs = render_env.reset() 
            renders = []
        
        max_steps = getattr(render_env, 'max_episode_steps', 1000)
        episode_frames = []
        
        for i in range(max_steps):
            with torch.no_grad(), autocast():
                obs_norm = train_state.normalizer(obs)
                action_dist, det_actions, _, _ = train_state.actor(obs_norm)
                if stochastic_eval:
                    actions = action_dist.sample()
                else:
                    actions = det_actions
            
            if env_type == "maniskill":
                next_obs, rewards, dones, _, infos = render_env.step(actions)
                
                # Track end-effector position if requested
                if track_trajectory and hasattr(render_env, 'unwrapped'):
                    try:
                        # Try to get end-effector position from the environment
                        # This works for most ManiSkill manipulation tasks
                        env = render_env.unwrapped.base_env
                        if hasattr(env, 'agent') and hasattr(env.agent, 'tcp'):
                            # Get TCP (Tool Center Point) position - this is the end-effector
                            ee_pos = env.agent.tcp.pose.p  # Shape: (num_envs, 3)
                            trajectories.append({
                                'step': i,
                                'ee_pos': ee_pos.cpu().numpy(),
                                'actions': actions.cpu().numpy(),
                                'rewards': rewards.cpu().numpy(),
                                'dones': dones.cpu().numpy(),
                            })
                    except Exception as e:
                        # If we can't get the end-effector position, skip tracking for this step
                        pass
                        
            elif env_type == "humanoid_bench":
                next_obs, rewards, dones, infos = render_env.step(actions)
                truncations = infos["time_outs"]
            else:
                next_obs, rewards, dones, _, infos = render_env.step(actions)
            
            if env_type == "mjx":
                try:
                    import jax.numpy as jnp
                    render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
                except (ImportError, AttributeError):
                    pass
            
            # Record frames for different environment types
            if env_type == "humanoid_bench":
                if i % 2 == 0:  # Record every 2nd frame to reduce video size
                    episode_frames.append(render_env.render())
            elif env_type == "maniskill":
                # For ManiSkill, try to get rgb_array if available
                if i % 2 == 0:
                    frame = render_env.render()
                    frame = frame.cpu().numpy() if frame is not None else None
                    if frame is not None:
                        # only take first env's frame
                        episode_frames.append(frame[0])
                # print ("ManiSkill rendering during rollout relies on built-in recording.")
            elif env_type == "mjx":
                if i % 2 == 0:
                    renders.append(render_env.state)
            
            if dones.any():
                break
            obs = next_obs
        
        # Process renders based on environment type
        if env_type == "mjx":
            try:
                episode_frames = render_env.render_trajectory(renders)
            except AttributeError:
                episode_frames = []
        elif env_type == "humanoid_bench":
            # episode_frames is already populated
            pass
        elif env_type == "maniskill":
            # For ManiSkill, if we have RecordEpisode wrapper, 
            # the video is automatically saved, but we return frames for wandb
            pass
        
        train_state.normalizer.train()
        
        # Return both frames and trajectories if tracking
        if track_trajectory:
            return episode_frames, trajectories
        return episode_frames
    
    return render_with_rollout


def configure_platform(cfg: DictConfig) -> DictConfig:
    cfg.platform.amp_enabled = (
        cfg.platform.amp_enabled and cfg.platform.cuda and torch.cuda.is_available()
    )
    cfg.platform.amp_device = (
        "cuda"
        if cfg.platform.cuda and torch.cuda.is_available()
        else "mps"
        if cfg.platform.cuda and torch.backends.mps.is_available()
        else "cpu"
    )
    return cfg


def save_checkpoint(
    cfg: DictConfig,
    train_state: TrainState,
    global_step: int,
    run_name: str,
) -> None:
    """Save model checkpoint to disk and optionally upload to wandb."""
    if cfg.checkpoint_dir is None:
        return
    
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"{run_name}_step_{global_step}.pt"
    
    checkpoint = {
        "global_step": global_step,
        "actor_state_dict": train_state.actor.state_dict(),
        "old_actor_state_dict": train_state.old_actor.state_dict(),
        "critic_state_dict": train_state.critic.state_dict(),
        "actor_optimizer_state_dict": train_state.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": train_state.critic_optimizer.state_dict(),
        "normalizer_state_dict": train_state.normalizer.state_dict() if hasattr(train_state.normalizer, 'state_dict') else None,
        "critic_normalizer_state_dict": train_state.critic_normalizer.state_dict() if hasattr(train_state.critic_normalizer, 'state_dict') else None,
        "scaler_state_dict": train_state.scaler.state_dict(),
        "config": OmegaConf.to_container(cfg),
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Also save the latest checkpoint
    latest_path = checkpoint_dir / f"{run_name}_latest.pt"
    torch.save(checkpoint, latest_path)
    print(f"Latest checkpoint saved to {latest_path}")
    
    # Upload to wandb if enabled
    if cfg.get("wandb_upload_checkpoints", False):
        try:
            # Create wandb artifact for the checkpoint
            artifact_name = f"{run_name}_checkpoint"
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"Model checkpoint at step {global_step}",
                metadata={
                    "global_step": global_step,
                    "run_name": run_name,
                    "env_name": cfg.env.name,
                    "algorithm": "reppo",
                }
            )
            
            # Add the checkpoint file to the artifact
            artifact.add_file(str(checkpoint_path), name=f"checkpoint_step_{global_step}.pt")
            
            # Also add the latest checkpoint
            artifact.add_file(str(latest_path), name="checkpoint_latest.pt")
            
            # Log the artifact
            wandb.log_artifact(artifact)
            print(f"Checkpoint uploaded to wandb as artifact: {artifact_name}")
            
        except Exception as e:
            print(f"Warning: Failed to upload checkpoint to wandb: {e}")
            # Don't fail the training if wandb upload fails


def load_checkpoint(
    cfg: DictConfig,
    train_state: TrainState,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[TrainState, int]:
    """Load model checkpoint from disk."""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    train_state.actor.load_state_dict(checkpoint["actor_state_dict"])
    train_state.old_actor.load_state_dict(checkpoint["old_actor_state_dict"])
    train_state.critic.load_state_dict(checkpoint["critic_state_dict"])
    train_state.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    train_state.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    
    if checkpoint.get("normalizer_state_dict") is not None and hasattr(train_state.normalizer, 'load_state_dict'):
        train_state.normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
    
    if checkpoint.get("critic_normalizer_state_dict") is not None and hasattr(train_state.critic_normalizer, 'load_state_dict'):
        train_state.critic_normalizer.load_state_dict(checkpoint["critic_normalizer_state_dict"])
    
    if checkpoint.get("scaler_state_dict") is not None:
        train_state.scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    global_step = checkpoint["global_step"]
    
    print(f"Checkpoint loaded successfully. Resuming from step {global_step}")
    
    return train_state, global_step


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="reppo_maniskill",
)
def main(cfg):
    cfg = configure_platform(cfg)
    cfg.hyperparameters = OmegaConf.merge(cfg.hyperparameters, cfg.experiment_overrides.hyperparameters)
    run_name = f"{cfg.name}_torch_{cfg.env.name}"

    # Add default checkpoint parameters if not present
    if not hasattr(cfg, 'checkpoint_dir'):
        cfg.checkpoint_dir = None
    if not hasattr(cfg, 'checkpoint_path'):
        cfg.checkpoint_path = None
    if not hasattr(cfg, 'save_checkpoint_interval'):
        cfg.save_checkpoint_interval = 0  # 0 means no periodic saving
    if not hasattr(cfg, 'save_final_checkpoint'):
        cfg.save_final_checkpoint = True
    if not hasattr(cfg, 'wandb_upload_checkpoints'):
        cfg.wandb_upload_checkpoints = False

    # Add eval-only parameter
    if not hasattr(cfg, 'eval_only'):
        cfg.eval_only = False
    if not hasattr(cfg, 'eval_episodes'):
        cfg.eval_episodes = 10  # Number of episodes to evaluate when eval_only=True

    # Add trajectory tracking parameters
    if not hasattr(cfg, 'save_trajectories'):
        cfg.save_trajectories = False  # Save end-effector trajectories during evaluation
    if not hasattr(cfg, 'trajectory_dir'):
        cfg.trajectory_dir = "./trajectories"  # Directory to save trajectory data and plots

    # Add default render parameters if not present
    if not hasattr(cfg.hyperparameters, 'render_interval'):
        cfg.hyperparameters.render_interval = 0  # 0 means no video recording
    if not hasattr(cfg.hyperparameters, 'render_fps'):
        cfg.hyperparameters.render_fps = 30
    if not hasattr(cfg, 'render_dir'):
        cfg.render_dir = None  # For ManiSkill local video saving

    # Validation for eval_only mode
    if cfg.eval_only and not cfg.checkpoint_path:
        raise ValueError("eval_only=True requires checkpoint_path to be specified")

    scaler = GradScaler(
        enabled=cfg.platform.amp_enabled and cfg.platform.amp_dtype == torch.float16
    )

    num_batches = cfg.hyperparameters.num_mini_batches
    batch_size = (
        cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps // num_batches
    )

    run_config = OmegaConf.to_container(cfg)
    run_config["method_name"] = "reppo"
    wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        config=run_config,
        save_code=True,
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.platform.torch_deterministic

    if not cfg.platform.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{cfg.platform.device_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{cfg.platform.device_rank}")
        else:
            raise ValueError("No GPU available")
    print(f"Using device: {device}")

    envs, eval_envs, render_env = make_envs(cfg=cfg, device=device, seed=cfg.seed)

    n_act = envs.num_actions
    n_obs = envs.num_obs if isinstance(envs.num_obs, int) else envs.num_obs[0]
    if envs.asymmetric_obs:
        n_critic_obs = (
            envs.num_privileged_obs
            if isinstance(envs.num_privileged_obs, int)
            else envs.num_privileged_obs[0]
        )
    else:
        n_critic_obs = n_obs

    if cfg.hyperparameters.normalize_env:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(
            shape=n_critic_obs, device=device
        )
    else:
        obs_normalizer = nn.Identity()
        critic_obs_normalizer = nn.Identity()

    actor = Actor(
        n_obs=n_obs,
        n_act=n_act,
        ent_start=cfg.hyperparameters.ent_start,
        kl_start=cfg.hyperparameters.kl_start,
        hidden_dim=cfg.hyperparameters.actor_hidden_dim,
        use_norm=cfg.hyperparameters.use_actor_norm,
        layers=cfg.hyperparameters.num_actor_layers,
        min_std=cfg.hyperparameters.actor_min_std,
        device=device,
    )
    old_actor = copy.deepcopy(actor)
    qnet = Critic(
        n_obs=n_critic_obs,
        n_act=n_act,
        num_atoms=cfg.hyperparameters.num_bins,
        vmin=cfg.hyperparameters.vmin,
        vmax=cfg.hyperparameters.vmax,
        hidden_dim=cfg.hyperparameters.critic_hidden_dim,
        use_norm=cfg.hyperparameters.use_critic_norm,
        use_encoder_norm=False,
        encoder_layers=cfg.hyperparameters.num_critic_encoder_layers,
        head_layers=cfg.hyperparameters.num_critic_head_layers,
        pred_layers=cfg.hyperparameters.num_critic_pred_layers,
        device=device,
    )

    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=torch.tensor(cfg.hyperparameters.lr, device=device),
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(cfg.hyperparameters.lr, device=device),
    )

    if envs.asymmetric_obs:
        (obs, *_), (critic_obs, *_) = envs.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        if cfg.env.type == "maniskill":
            (obs, *_) = envs.reset()
            critic_obs = obs
        else:
            obs = envs.reset()
            critic_obs = obs

    train_state = TrainState(
        obs=obs,
        critic_obs=critic_obs,
        actor=actor,
        old_actor=old_actor,
        critic=qnet,
        normalizer=obs_normalizer,
        critic_normalizer=critic_obs_normalizer,
        actor_optimizer=actor_optimizer,
        critic_optimizer=q_optimizer,
        device=device,
        scaler=scaler,
    )

    print(
        summary(
            train_state.critic,
            input_data=(critic_obs, torch.zeros((cfg.hyperparameters.num_envs, n_act), device=device)),
            depth=10,
        )
    )
    print(summary(train_state.actor, input_data=(obs,), depth=10))
    # create functions
    collect_fn = make_collect_fn(cfg, envs, cfg.env.type)
    postprocess_fn = make_postprocess_fn(cfg, envs)
    update_critic = make_critic_update_fn(cfg, train_state)
    update_actor = make_actor_update_fn(cfg, train_state)
    evaluate = make_evaluate_fn(cfg, eval_envs, cfg.env.type)
    render_rollout = make_render_fn(cfg, render_env, cfg.env.type)

    if cfg.platform.compile:
        mode = "max-autotune-no-cudagraphs"
        update_critic = torch.compile(update_critic, mode=mode)
        update_actor = torch.compile(update_actor, mode=mode)
        postprocess_fn = torch.compile(postprocess_fn, mode=mode)
        train_state.compile()

    # Load checkpoint if specified
    if cfg.checkpoint_path:
        train_state, global_step = load_checkpoint(cfg, train_state, cfg.checkpoint_path, device)
    else:
        global_step = 0

    # If eval_only mode, run evaluation and exit
    if cfg.eval_only:
        print(f"Running evaluation only mode with {cfg.eval_episodes} episodes...")

        eval_returns = []
        eval_lengths = []
        eval_successes = []
        all_trajectories = []

        stochastic_eval = cfg.env.get("stochastic_eval", False)
        track_traj = cfg.save_trajectories and cfg.env.type == "maniskill"

        if track_traj:
            print(f"Trajectory tracking enabled. Will save to {cfg.trajectory_dir}")

        for episode in range(cfg.eval_episodes):
            print(f"Running evaluation episode {episode + 1}/{cfg.eval_episodes}")

            eval_avg_return, eval_avg_length, eval_info = evaluate(
                train_state, 
                stochastic_eval=stochastic_eval
            )

            eval_returns.append(eval_avg_return)
            eval_lengths.append(eval_avg_length)

            # Extract success rate if available
            success_rate = eval_info.get('success', 0.0)
            if isinstance(success_rate, torch.Tensor):
                success_rate = success_rate.item()
            elif isinstance(success_rate, np.ndarray):
                success_rate = success_rate.mean()
            eval_successes.append(success_rate)

            print(f"Episode {episode + 1}: Return={eval_avg_return:.2f}, Length={eval_avg_length:.2f}, Success={success_rate:.2f}")

            # Record video and/or track trajectory
            if (cfg.hyperparameters.render_interval > 0) or track_traj:
                if track_traj:
                    print(f"Recording video and tracking trajectory for episode {episode + 1}")
                else:
                    print(f"Recording video for episode {episode + 1}")
                try:
                    # Call render_rollout with trajectory tracking if enabled
                    result = render_rollout(train_state, stochastic_eval=stochastic_eval, track_trajectory=track_traj)

                    # Unpack result based on whether trajectory tracking is enabled
                    if track_traj:
                        renders, trajectories = result
                        if trajectories:
                            all_trajectories.append(trajectories)
                            print(f"  Captured {len(trajectories)} trajectory steps")
                    else:
                        renders = result
                    if renders and len(renders) > 0:
                        if cfg.env.type == "humanoid_bench":
                            video_array = np.array(renders)
                            if video_array.ndim == 4:  # (T, H, W, C)
                                video_array = video_array.transpose(0, 3, 1, 2)
                            render_video = wandb.Video(
                                video_array,
                                fps=cfg.hyperparameters.render_fps,
                                format="gif",
                            )
                            wandb.log({f"eval_video_episode_{episode + 1}": render_video})
                            print(f"Video recorded with {len(renders)} frames")
                        else:
                            print("Video saved locally (if recording enabled)")
                except Exception as e:
                    print(f"Error recording video for episode {episode + 1}: {e}")

        # Calculate and log final statistics
        mean_return = np.mean(eval_returns)
        std_return = np.std(eval_returns)
        mean_length = np.mean(eval_lengths)
        std_length = np.std(eval_lengths)
        mean_success = np.mean(eval_successes)
        std_success = np.std(eval_successes)

        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Episodes evaluated: {cfg.eval_episodes}")
        print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")
        print(f"Mean length: {mean_length:.2f} ± {std_length:.2f}")
        print(f"Mean success: {mean_success:.2f} ± {std_success:.2f}")
        print("="*50)

        # Log summary to wandb
        wandb.log({
            "eval_summary/mean_return": mean_return,
            "eval_summary/std_return": std_return,
            "eval_summary/mean_length": mean_length,
            "eval_summary/std_length": std_length,
            "eval_summary/mean_success": mean_success,
            "eval_summary/std_success": std_success,
            "eval_summary/episodes": cfg.eval_episodes,
        })

        # Save and plot trajectories if enabled
        if cfg.save_trajectories and all_trajectories:
            print("\n" + "="*50)
            print("SAVING TRAJECTORIES")
            print("="*50)

            episode_info = {
                'returns': eval_returns,
                'lengths': eval_lengths,
                'successes': eval_successes,
                'mean_return': mean_return,
                'std_return': std_return,
                'mean_success': mean_success,
            }

            try:
                plot_enabled = cfg.get('plot_trajectories', True)
                traj_file, plot_file = save_and_plot_trajectories(
                    all_trajectories=all_trajectories,
                    save_dir=cfg.trajectory_dir,
                    env_name=cfg.env.name,
                    episode_info=episode_info,
                    plot_trajectories=plot_enabled,
                )

                # Log trajectory plot to wandb (only if plotting was enabled)
                if plot_file:
                    try:
                        import matplotlib.pyplot as plt
                        trajectory_image = wandb.Image(str(plot_file))
                        wandb.log({"eval_trajectories/summary_plot": trajectory_image})
                        print(f"Uploaded trajectory plot to wandb")
                    except Exception as e:
                        print(f"Could not upload trajectory plot to wandb: {e}")

            except Exception as e:
                print(f"Error saving trajectories: {e}")
                import traceback
                traceback.print_exc()

            print("="*50)

        print("Evaluation completed. Exiting.")
        return

    total_env_steps = (
        cfg.hyperparameters.total_time_steps
        // (cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps)
        + 1
    )

    pbar = tqdm.tqdm(total=cfg.hyperparameters.total_time_steps, initial=global_step)
    start_time = None
    desc = ""

    eval_interval = total_env_steps // cfg.hyperparameters.num_eval
    stochastic_eval = cfg.env.get("stochastic_eval", False)

    while global_step < total_env_steps:
        if start_time is None and global_step >= cfg.measure_burnin:
            start_time = time.time()
            measure_burnin = global_step

        train_state, transition, infos = collect_fn(train_state)
        data = postprocess_fn(train_state, transition)

        for _ in range(cfg.hyperparameters.num_epochs):
            indices = torch.randperm(
                cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps,
                device=device,
            )
            data = data[indices].contiguous()
            for j in range(num_batches):
                mini_batch = data[j * batch_size : (j + 1) * batch_size]
                critic_logs_dict = update_critic(mini_batch)
                actor_logs_dict = update_actor(mini_batch)
                logs_dict = {
                    **critic_logs_dict,
                    **actor_logs_dict,
                }

        for param, target_param in zip(actor.parameters(), old_actor.parameters()):
            target_param.data.copy_(param.data)
        if start_time is not None:
            # @TODO: shouldn't that be env_steps per second?
            speed = (
                cfg.hyperparameters.num_envs
                * cfg.hyperparameters.num_steps
                * (global_step - measure_burnin)
                / (time.time() - start_time)
            )
            pbar.set_description(f"{speed: 4.4f} sps, " + desc)
            with torch.inference_mode():
                logs = {
                    "critic/qf_loss": logs_dict["qf_loss"].mean(),
                    "critic/qf_max": logs_dict["qf_max"].max(),
                    "critic/qf_min": logs_dict["qf_min"].min(),
                    "critic/qf_mean": logs_dict["qf_mean"].mean(),
                    "critic/embedding_loss": logs_dict["embedding_loss"].mean(),
                    "critic/critic_grad_norm": logs_dict["critic_grad_norm"].mean(),
                    "actor/actor_loss": logs_dict["actor_loss"].mean(),
                    "actor/actor_grad_norm": logs_dict["actor_grad_norm"].mean(),
                    "actor/kl": logs_dict["kl"].mean(),
                    "actor/entropy": logs_dict["entropy"].mean(),
                    "actor/temperature": logs_dict["temperature"].mean(),
                    "actor/lagrangian": logs_dict["lagrangian"].mean(),
                    "actor/entropy_loss": logs_dict["entropy_loss"].mean(),
                    "actor/lagrangian_loss": logs_dict["lagrangian_loss"].mean(),
                    "train/rewards_batch": data["rewards"].mean(),
                }

                if cfg.env.type == "maniskill":
                    logs.update(
                        {
                            "train/return": torch.stack(
                                [info["log_info"]["return"] for info in infos]
                            ).mean(),
                            "train/episode_len": torch.stack(
                                [info["log_info"]["episode_len"] for info in infos]
                            )
                            .float()
                            .mean(),
                            "train/success": torch.stack(
                                [info["log_info"]["success"] for info in infos]
                            )
                            .float()
                            .mean(),
                        }
                    )

                if eval_interval > 0 and global_step % eval_interval == 0:
                    print(f"Evaluating at global step {global_step}")
                    if stochastic_eval:
                        eval_avg_return, eval_avg_length, stoch_eval_info = evaluate(
                            train_state, stochastic_eval=stochastic_eval
                        )
                        eval_avg_return, eval_avg_length, eval_info = evaluate(
                            train_state
                        )
                        eval_info = {
                            **eval_info,
                            **{f"stoch/{k}": v for k, v in stoch_eval_info.items()},
                        }
                    else:
                        eval_avg_return, eval_avg_length, eval_info = evaluate(
                            train_state
                        )
                    if cfg.env.type in [
                        "humanoid_bench",
                        "isaaclab",
                        "mtbench",
                    ]:
                        # NOTE: Hacky way of evaluating performance, but just works
                        obs, *_ = envs.reset()
                    logs["eval/avg_return"] = eval_avg_return
                    logs["eval/avg_length"] = eval_avg_length
                    for key, value in eval_info.items():
                        if isinstance(value, torch.Tensor):
                            logs[f"eval/{key}"] = value.mean().item()
                        elif isinstance(value, np.ndarray):
                            logs[f"eval/{key}"] = value.mean()
                        else:
                            logs[f"eval/{key}"] = value
                    print(
                        f"Eval return: {eval_avg_return:.2f}, length: {eval_avg_length:.2f}, env steps: {global_step * cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps} success rate: {eval_info.get('success', 0.0):.2f}"
                    )

                # Record video if render_interval is set
                if (cfg.hyperparameters.render_interval > 0 and 
                    global_step % cfg.hyperparameters.render_interval == 0):
                    print(f"Recording video at global step {global_step}")
                    try:
                        renders = render_rollout(train_state)
                        if renders and len(renders) > 0:
                            # Convert renders to numpy array with correct format for wandb
                            if cfg.env.type == "humanoid_bench":
                                # renders is a list of (H, W, C) arrays
                                video_array = np.array(renders)
                                # Convert to (T, C, H, W) format for wandb
                                if video_array.ndim == 4:  # (T, H, W, C)
                                    video_array = video_array.transpose(0, 3, 1, 2)

                                render_video = wandb.Video(
                                    video_array,
                                    fps=cfg.hyperparameters.render_fps,
                                    format="gif",
                                )
                                logs["render_video"] = render_video
                                print(f"Video recorded with {len(renders)} frames")

                            elif cfg.env.type == "maniskill":
                                # For ManiSkill environments
                                if isinstance(renders, list) and len(renders) > 0:
                                    video_array = np.array(renders)
                                    if video_array.ndim == 4:  # (T, H, W, C)
                                        video_array = video_array.transpose(0, 3, 1, 2)
                                    elif video_array.ndim == 3:  # Single frame (H, W, C)
                                        video_array = video_array.transpose(2, 0, 1)[None, ...]  # (1, C, H, W)

                                    render_video = wandb.Video(
                                        video_array,
                                        fps=cfg.hyperparameters.render_fps,
                                        format="gif",
                                    )
                                    logs["render_video"] = render_video
                                    print(f"Video recorded with {len(renders)} frames")
                                else:
                                    print("ManiSkill video saved locally (if RecordEpisode wrapper enabled)")

                            elif cfg.env.type == "mjx":
                                # For MJX environments, renders might be in a different format
                                if hasattr(renders, 'shape') and len(renders) > 0:
                                    video_array = np.array(renders)
                                    if video_array.ndim == 4:
                                        video_array = video_array.transpose(0, 3, 1, 2)
                                    render_video = wandb.Video(
                                        video_array,
                                        fps=cfg.hyperparameters.render_fps,
                                        format="gif",
                                    )
                                    logs["render_video"] = render_video
                                    print(f"Video recorded with {len(renders)} frames")
                        else:
                            print("No frames rendered - skipping video logging")
                            if cfg.env.type == "maniskill" and cfg.render_dir:
                                print(f"Check {cfg.render_dir} for locally saved videos")
                    except Exception as e:
                        print(f"Error recording video: {e}")

                # Save periodic checkpoint
                if (cfg.save_checkpoint_interval > 0 and 
                    global_step % cfg.save_checkpoint_interval == 0):
                    save_checkpoint(cfg, train_state, global_step, run_name)

            wandb.log(
                {
                    "speed": speed,
                    "frame": global_step * cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps,
                    **logs,
                },
                step=global_step * cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps,
            )

        global_step += 1
        pbar.update(n=cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps)

    # Save final checkpoint
    if cfg.save_final_checkpoint:
        save_checkpoint(cfg, train_state, global_step, run_name)
        print("Training completed. Final checkpoint saved.")


if __name__ == "__main__":
    main()
