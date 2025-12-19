import logging
import time
import typing
from typing import Callable, Any

import hydra
import jax
import numpy as np
import optax
import optuna
from flax import nnx, struct
from flax.struct import PyTreeNode
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax import numpy as jnp
from jax.random import PRNGKey
from omegaconf import DictConfig, OmegaConf

import wandb
from src.env_utils.jax_wrappers import (
    BraxGymnaxWrapper,
    TanhClipAction,
    LogWrapper,
    MjxGymnaxWrapper,
    MjxDiffEnvWrapper,
    DiffNormalizeVec,
)
from src.jaxrl import utils
from src.networks.diffusion.models import ControlNetwork
from src.networks.jax_models_DMERL import (
    CategoricalCriticNetwork,
    CriticNetwork,
    DiffusionModel,
    DMERLActor,
    sde_integrator,
    ode_integrator,
    logratio_DIME as logratio,
)

logging.basicConfig(level=logging.INFO)


class Policy(typing.Protocol):
    def __call__(
        self,
        key: jax.random.PRNGKey,
        obs: PyTreeNode,
    ) -> tuple[PyTreeNode, PyTreeNode]:
        pass


class Transition(struct.PyTreeNode):
    obs: jax.Array
    critic_obs: jax.Array
    action: jax.Array
    reward: jax.Array
    soft_reward: jax.Array
    next_emb: jax.Array
    value: jax.Array
    done: jax.Array
    truncated: jax.Array
    importance_weight: jax.Array
    info: dict[str, jax.Array]


class ReppoConfig(struct.PyTreeNode):
    lr: float
    gamma: float
    total_time_steps: int
    num_steps: int
    lmbda: float
    lmbda_min: float
    num_mini_batches: int
    num_envs: int
    num_epochs: int
    max_grad_norm: float | None
    normalize_env: bool
    polyak: float
    exploration_noise_min: float
    exploration_noise_max: float
    exploration_base_envs: int
    kl_action_rep: int
    ent_start: float
    ent_target_mult: float
    kl_start: float
    action_clip_value: float = 1.0
    env_action_clip_value: float = 1.0
    eval_interval: int = 10
    num_eval: int = 25
    max_episode_steps: int = 1000
    critic_hidden_dim: int = 512
    actor_hidden_dim: int = 512
    vmin: int = -100
    vmax: int = 100
    num_bins: int = 250
    hl_gauss: bool = False
    kl_bound: float = 1.0
    aux_loss_mult: float = 0.0
    update_kl_lagrangian: bool = True
    update_entropy_lagrangian: bool = True
    use_critic_norm: bool = True
    num_critic_encoder_layers: int = 1
    num_critic_head_layers: int = 1
    num_critic_pred_layers: int = 1
    use_simplical_embedding: bool = False
    use_critic_skip: bool = False
    use_actor_norm: bool = True
    num_actor_layers: int = 2
    actor_min_std: float = 0.05
    use_actor_skip: bool = False
    reduce_kl: bool = True
    reverse_kl: bool = False
    anneal_lr: bool = False
    actor_kl_clip_mode: str = "clipped"
    use_lax_scan: bool = True

    # diffusion settings
    diffusion: Any = None # DictConfig
    ode_coefs: list = None

class SACTrainState(struct.PyTreeNode):
    critic: nnx.TrainState
    actor: nnx.TrainState
    actor_target: nnx.TrainState
    iteration: int
    time_steps: int
    last_env_state: EnvState
    last_obs: jax.Array
    last_critic_obs: jax.Array

def maybe_lax_scan(
    f: Callable,
    init,
    xs=None,
    *,
    length: int | None = None,
    reverse: bool = False,
    use_scan: bool = True,
):
    """Run jax.lax.scan or an equivalent Python loop when debugging."""
    if use_scan:
        return jax.lax.scan(f, init, xs, length=length, reverse=reverse)

    if xs is None:
        if length is None:
            raise ValueError("length must be provided when xs is None.")
        xs_sequence = [None] * length
    else:
        leaves, _ = jax.tree_util.tree_flatten(xs)
        if not leaves:
            raise ValueError("scan inputs must contain at least one leaf.")
        first_shape = np.shape(leaves[0])
        if len(first_shape) == 0:
            raise ValueError("scan inputs must have a leading dimension.")
        seq_length = int(first_shape[0])

        def take_index(i):
            return jax.tree.map(lambda arr, idx=i: arr[idx], xs)

        xs_sequence = [take_index(i) for i in range(seq_length)]

    if reverse:
        xs_iter = reversed(xs_sequence)
    else:
        xs_iter = xs_sequence

    carry = init
    outputs = []
    has_output = None
    for x in xs_iter:
        carry, y = f(carry, x)
        if has_output is None:
            has_output = y is not None
        if has_output:
            outputs.append(y)

    if has_output:
        if reverse:
            outputs = list(reversed(outputs))
        stacked = jax.tree.map(lambda *vals: jnp.stack(vals), *outputs)
    else:
        stacked = None
    return carry, stacked

def make_sde_eval_fn(
    env: Environment, max_episode_steps: int, reward_scale: float = 1.0
) -> Callable[
    [jax.random.PRNGKey, SACTrainState, PyTreeNode | None], dict[str, float]
]:
    """
    Creates a static evaluation function for SDE (stochastic) policy.
    This will be JIT-compiled "lean" with only the sde_integrator path.
    """
    def sde_evaluation_fn(
        key: jax.random.PRNGKey,
        train_state: SACTrainState,
        norm_state: PyTreeNode | None,
    ):
        actor_model = nnx.merge(
            train_state.actor.graphdef, train_state.actor.params
        )

        # --- Policy is hard-coded to actor_model.sample() ---
        def sde_policy(key: PRNGKey, obs: jax.Array) -> tuple[jax.Array, dict]:
            action, *_ = actor_model.sample(key, obs)
            return action, {}

        def step_env(carry, _):
            key, env_state, obs = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action, _ = sde_policy(act_key, obs) # <-- Calls SDE path
            
            step_key = jax.random.split(env_key, env.num_envs)
            obs, _, env_state, reward, done, info = env.step(
                step_key, env_state, action
            )
            return (key, env_state, obs), info

        key, init_key = jax.random.split(key)
        init_key = jax.random.split(init_key, env.num_envs)
        obs, _, env_state = env.reset(init_key, norm_state)
        
        jax.debug.print("max episode steps for sde evaluation: {}", max_episode_steps)
        key, env_key = jax.random.split(key)
        _, infos = jax.lax.scan(
            f=step_env,
            init=(key, env_state, obs),
            xs=None,
            length=max_episode_steps,
        )

        return { # ... (return metrics dict as before)
            "episode_return": infos["returned_episode_returns"].mean(
                where=infos["returned_episode"]
            )
            * reward_scale,
            "episode_return_std": infos["returned_episode_returns"].std(
                where=infos["returned_episode"]
            ),
            "episode_length": infos["returned_episode_lengths"].mean(
                where=infos["returned_episode"]
            ),
            "episode_length_std": infos["returned_episode_lengths"].std(
                where=infos["returned_episode"]
            ),
            "num_episodes": infos["returned_episode"].sum(),
        }

    return sde_evaluation_fn


def make_ode_eval_fn(
    env: Environment, max_episode_steps: int, reward_scale: float = 1.0
) -> Callable[
    [jax.random.PRNGKey, SACTrainState, float, PyTreeNode | None], dict[str, float]
]:
    """
    Creates a static evaluation function for ODE (deterministic) policy.
    This will be JIT-compiled "lean" with only the ode_integrator path.
    """
    def ode_evaluation_fn(
        key: jax.random.PRNGKey,
        train_state: SACTrainState,
        ode_coef: float,           # <-- Takes ode_coef as an argument
        norm_state: PyTreeNode | None,
    ):
        actor_model = nnx.merge(
            train_state.actor.graphdef, train_state.actor.params
        )

        # --- Policy is hard-coded to actor_model.det_action() ---
        def ode_policy(key: PRNGKey, obs: jax.Array) -> tuple[jax.Array, dict]:
            action, *_ = actor_model.det_action(
                key, obs, ode=True, ode_coef=ode_coef
            )
            return action, {}

        def step_env(carry, _):
            key, env_state, obs = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action, _ = ode_policy(act_key, obs) # <-- Calls ODE path
            
            step_key = jax.random.split(env_key, env.num_envs)
            obs, _, env_state, reward, done, info = env.step(
                step_key, env_state, action
            )
            return (key, env_state, obs), info

        key, init_key = jax.random.split(key)
        init_key = jax.random.split(init_key, env.num_envs)
        obs, _, env_state = env.reset(init_key, norm_state)
        
        jax.debug.print("max episode steps for ode evaluation: {}", max_episode_steps)
        key, env_key = jax.random.split(key)
        _, infos = jax.lax.scan(
            f=step_env,
            init=(key, env_state, obs),
            xs=None,
            length=max_episode_steps,
        )

        return { # ... (return metrics dict as before)
            "episode_return": infos["returned_episode_returns"].mean(
                where=infos["returned_episode"]
            )
            * reward_scale,
            "episode_return_std": infos["returned_episode_returns"].std(
                where=infos["returned_episode"]
            ),
            "episode_length": infos["returned_episode_lengths"].mean(
                where=infos["returned_episode"]
            ),
            "episode_length_std": infos["returned_episode_lengths"].std(
                where=infos["returned_episode"]
            ),
            "num_episodes": infos["returned_episode"].sum(),
        }

    return ode_evaluation_fn

def make_init(
    cfg: ReppoConfig,
    env: Environment,
    env_params: EnvParams = None,
) -> Callable[[jax.Array], SACTrainState]:
    def init(key: jax.random.PRNGKey) -> SACTrainState:
        # Number of calls to train_step
        key, model_key = jax.random.split(key)
        obs_dim, critic_obs_dim = env.get_obs_space_sizes()
        action_dim=env.action_space(env_params).shape[0]
        
        # DIME initialize scheduler
        dt_schedule = hydra.utils.call(cfg.diffusion.dt_schedule)

        if cfg.diffusion.learn_forward:
            forward_model: nnx.Module = ControlNetwork(
                action_dim=action_dim,
                observation_dim=obs_dim,
                num_layers=cfg.diffusion.score_model.num_layers,
                num_hid=cfg.diffusion.score_model.num_hid,
                num_time_hid=cfg.diffusion.score_model.num_time_hid,
                num_time_out=cfg.diffusion.score_model.num_time_out,
                outer_clip=cfg.diffusion.score_model.outer_clip,
                inner_clip=cfg.diffusion.score_model.inner_clip,
                weight_init=cfg.diffusion.score_model.weight_init,
                bias_init=cfg.diffusion.score_model.bias_init,
                layer_norm=cfg.diffusion.score_model.layer_norm,
                layer_norm_type=cfg.diffusion.score_model.layer_norm_type,
                rngs=nnx.Rngs(model_key),
            )
        else:
            forward_model = None

        if cfg.diffusion.learn_backward:
            backward_model: nnx.Module = ControlNetwork(
                action_dim=action_dim,
                observation_dim=obs_dim,
                num_layers=cfg.diffusion.score_model.num_layers,
                num_hid=cfg.diffusion.score_model.num_hid,
                num_time_hid=cfg.diffusion.score_model.num_time_hid,
                num_time_out=cfg.diffusion.score_model.num_time_out,
                outer_clip=cfg.diffusion.score_model.outer_clip,
                inner_clip=cfg.diffusion.score_model.inner_clip,
                weight_init=cfg.diffusion.score_model.weight_init,
                bias_init=cfg.diffusion.score_model.bias_init,
                layer_norm=cfg.diffusion.score_model.layer_norm,
                layer_norm_type=cfg.diffusion.score_model.layer_norm_type,
                rngs=nnx.Rngs(model_key),
            )
        else:
            backward_model = None

        diffusion_model = DiffusionModel(
            action_dim=action_dim,
            observation_dim=obs_dim,
            fwd_model=forward_model,
            bwd_model=backward_model,
            diff_steps=cfg.diffusion.diff_steps,
            init_std=cfg.diffusion.init_std,
            friction=cfg.diffusion.friction,
            per_dim_friction=cfg.diffusion.per_dim_friction,
            dt=cfg.diffusion.dt,
            learn_dt=cfg.diffusion.learn_dt,
            per_step_dt=cfg.diffusion.per_step_dt,
            learn_prior=cfg.diffusion.learn_prior,
            learn_betas=cfg.diffusion.learn_betas,
            learn_friction=cfg.diffusion.learn_friction,
            learn_mass_matrix=cfg.diffusion.learn_mass_matrix,
            dt_schedule=dt_schedule,
            rngs=nnx.Rngs(model_key),
        )

        actor_networks = DMERLActor(
            action_dim=action_dim,
            observation_dim=obs_dim,
            diffusion_model=diffusion_model,
            logratio=logratio,
            kl_start=cfg.kl_start,
            ent_start=cfg.ent_start,
            sde_integrator=sde_integrator,
            ode_integrator=ode_integrator,
            action_clip_value=cfg.action_clip_value,
        )

        actor_target_networks = DMERLActor(
            action_dim=action_dim,
            observation_dim=obs_dim,
            diffusion_model=diffusion_model,
            logratio=logratio,
            kl_start=cfg.kl_start,
            ent_start=cfg.ent_start,
            sde_integrator=sde_integrator,
            ode_integrator=ode_integrator,
        )

        if cfg.hl_gauss:
            critic_networks: nnx.Module = CategoricalCriticNetwork(
                obs_dim=critic_obs_dim,
                action_dim=action_dim,
                hidden_dim=cfg.critic_hidden_dim,
                num_bins=cfg.num_bins,
                vmin=cfg.vmin,
                vmax=cfg.vmax,
                num_time_hid = cfg.diffusion.score_model.num_time_hid,
                num_time_out = cfg.diffusion.score_model.num_time_out,
                use_norm=cfg.use_critic_norm,
                encoder_layers=cfg.num_critic_encoder_layers,
                use_simplical_embedding=cfg.use_simplical_embedding,
                head_layers=cfg.num_critic_head_layers,
                pred_layers=cfg.num_critic_pred_layers,
                use_skip=cfg.use_critic_skip,
                rngs=nnx.Rngs(model_key),
            )
        else:
            critic_networks: nnx.Module = CriticNetwork(
                obs_dim=critic_obs_dim,
                action_dim=action_dim,
                hidden_dim=cfg.critic_hidden_dim,
                use_norm=cfg.use_critic_norm,
                encoder_layers=cfg.num_critic_encoder_layers,
                use_simplical_embedding=cfg.use_simplical_embedding,
                head_layers=cfg.num_critic_head_layers,
                pred_layers=cfg.num_critic_pred_layers,
                use_skip=cfg.use_critic_skip,
                rngs=nnx.Rngs(model_key),
            )

        if not cfg.anneal_lr:
            lr = cfg.lr
        else:
            num_iterations = cfg.total_time_steps // cfg.num_steps // cfg.num_envs
            num_updates = num_iterations * cfg.num_epochs * cfg.num_mini_batches
            lr = optax.linear_schedule(cfg.lr, 0, num_updates)

        print(f"Using learning rate: {lr}")
        if cfg.max_grad_norm is not None:
            actor_optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.max_grad_norm),
                optax.adam(lr)
            )
            critic_optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.max_grad_norm),
                optax.adam(lr)
            )
        else:
            actor_optimizer = optax.adam(lr)
            critic_optimizer = optax.adam(lr)

        actor_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(actor_networks),
            params=nnx.state(actor_networks),
            tx=actor_optimizer,
        )
        actor_target_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(actor_target_networks),
            params=nnx.state(actor_target_networks),
            tx=optax.set_to_zero(),
        )
        critic_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(critic_networks),
            params=nnx.state(critic_networks),
            tx=critic_optimizer,
        )

        actor_param_count = utils.count_params(actor_trainstate.params)
        critic_param_count = utils.count_params(critic_trainstate.params)
        
        print(f"Actor parameters: {actor_param_count:,}")
        print(f"Critic parameters: {critic_param_count:,}")
        print(f"Total parameters: {actor_param_count + critic_param_count:,}")

        key, env_key = jax.random.split(key)
        env_key = jax.random.split(env_key, cfg.num_envs)
        obs, critic_obs, env_state = env.reset(key=env_key, params=env_params)

        # randomize initial time step to prevent all envs stepping in tandem
        jax.debug.print("Randomizing initial steps with max {}", cfg.max_episode_steps * cfg.diffusion.diff_steps)
        _env_state = env_state.unwrapped()
        jax.debug.print("_env_state.info[\"steps\"].shape: {}", _env_state.info["steps"].shape)
        key, randomize_steps_key = jax.random.split(key)
        _env_state.info["steps"] = jax.random.randint(
            randomize_steps_key,
            _env_state.info["steps"].shape,
            0,
            cfg.max_episode_steps,# * cfg.diffusion.diff_steps,
        ).astype(jnp.float32)
        env_state.set_env_state(_env_state)

        return SACTrainState(
            actor=actor_trainstate,
            actor_target=actor_target_trainstate,
            critic=critic_trainstate,
            iteration=0,
            time_steps=0,
            last_env_state=env_state,
            last_obs=obs,
            last_critic_obs=critic_obs,
        )

    return init


def make_train_fn(
    cfg: ReppoConfig,
    env: Environment,
    env_params: EnvParams = None,
    log_callback: Callable[[SACTrainState, dict[str, jax.Array]], None] | None = None,
    num_seeds: int = 1,
    reward_scale: float = 1.0,
):
    """
    Create training function with support for evaluating different ODE coefficients.
    
    Args:
        cfg: Configuration
        env: Environment
        env_params: Environment parameters
        log_callback: Logging callback
        num_seeds: Number of seeds
        reward_scale: Reward scaling
    """
    diff_steps = getattr(cfg.diffusion, "diff_steps", None)
    if diff_steps is not None and diff_steps > 0:
        #   adjusted = cfg.vmax * (1.0 / diff_steps)
        #   cfg = cfg.replace(vmax=adjusted)

        #   adjusted = cfg.vmin * (1.0 / diff_steps)
        #   cfg = cfg.replace(vmin=adjusted)

    #     adjusted_gamma = cfg.gamma ** (1.0 / diff_steps)
    #     cfg = cfg.replace(gamma=adjusted_gamma)

    #     adjusted_lambda = cfg.lmbda ** (1.0 / diff_steps)
    #     cfg = cfg.replace(lmbda=adjusted_lambda)

    #     # adjusted_total_time_steps = cfg.total_time_steps * diff_steps
    #     # cfg = cfg.replace(total_time_steps=adjusted_total_time_steps)

        # adjusted_num_steps = cfg.num_steps * diff_steps
        # cfg = cfg.replace(num_steps=adjusted_num_steps)
        pass


    env = LogWrapper(env, cfg.num_envs)
    env = TanhClipAction(env)
    # env = VecEnv(env, cfg.num_envs)
    if cfg.normalize_env:
        env = DiffNormalizeVec(env)

    # eval_fn = make_eval_fn(env, cfg.max_episode_steps, reward_scale=reward_scale)
    eval_env_steps = cfg.max_episode_steps #* cfg.diffusion.diff_steps
    sde_eval_fn = make_sde_eval_fn(env, eval_env_steps, reward_scale=reward_scale)
    ode_eval_fn = make_ode_eval_fn(env, eval_env_steps, reward_scale=reward_scale)
    action_size_target = (
        jnp.prod(jnp.array(env.action_space(env_params).shape)) * cfg.ent_target_mult
    )

    def collect_rollout(
        key: PRNGKey, train_state: SACTrainState
    ) -> tuple[Transition, SACTrainState]:
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)

        offset = (
            jnp.arange(cfg.num_envs - cfg.exploration_base_envs)[:, None]
            * (cfg.exploration_noise_max - cfg.exploration_noise_min)
            / (cfg.num_envs - cfg.exploration_base_envs)
        ) + cfg.exploration_noise_min
        offset = jnp.concatenate(
            [
                jnp.ones((cfg.exploration_base_envs, 1)) * cfg.exploration_noise_min,
                offset,
            ],
            axis=0,
        )

        def step_env(carry, _) -> tuple[tuple, Transition]:
            key, env_state, train_state, obs, critic_obs = carry
            key, act_key, step_key = jax.random.split(key, 3)

            step_key = jax.random.split(step_key, cfg.num_envs)

            # get policy action
            action, gen_log_prob, dest_log_prob = actor_model.vmap_sample_next_step(obs, act_key) 
            action = jax.lax.stop_gradient(action) ### stop grad because it is then used in next_obs
            
            next_obs, next_critic_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, action
            )            
            importance_weight = jnp.zeros((cfg.num_envs,))

            # compute next state embedding and value
            key, next_act_key = jax.random.split(key)
            next_action, next_gen_log_prob, next_dest_log_prob = actor_model.vmap_sample_next_step(next_obs, next_act_key)
            next_action = jax.lax.stop_gradient(next_action) 
            # compute next state embedding and value
            next_emb, _, _, value = critic_model.forward(next_critic_obs, next_action)
            log_ratio = next_gen_log_prob - next_dest_log_prob
            log_ratio = jax.lax.stop_gradient(log_ratio)
            ### print with jax debug the reward and the log ratio

            soft_reward = (
                reward
                - cfg.gamma * log_ratio.squeeze() * actor_model.temperature()
            )
            # print truncated, done and obs
            # jax.debug.print("truncated: {t}, done: {d}", t=next_env_state.truncated, d=done)
            # jax.debug.print("obs: {o}", o=obs)
            transition = Transition(
                obs=obs,
                critic_obs=critic_obs,
                action=action,
                next_emb=next_emb,
                reward=reward,
                soft_reward=soft_reward,
                value=value,
                done=done,
                truncated=next_env_state.truncated,
                info=info,
                importance_weight=importance_weight,
            )
            return (
                key,
                next_env_state,
                train_state,
                next_obs,
                next_critic_obs,
            ), transition

        jax.debug.print("Collecting rollout of {} steps", cfg.num_steps)
        rollout_state, transitions = maybe_lax_scan(
            f=step_env,
            init=(
                key,
                train_state.last_env_state,
                train_state,
                train_state.last_obs,
                train_state.last_critic_obs,
            ),
            length=cfg.num_steps,
            use_scan=cfg.use_lax_scan,
        )
        _, last_env_state, train_state, last_obs, last_critic_obs = rollout_state
        train_state = train_state.replace(
            last_env_state=last_env_state,
            last_obs=last_obs,
            last_critic_obs=last_critic_obs,
            time_steps=train_state.time_steps + cfg.num_steps * cfg.num_envs,
        )

        return transitions, train_state

    def learn_step(
        key: PRNGKey, train_state: SACTrainState, batch: Transition
    ) -> tuple[SACTrainState, dict[str, jax.Array]]:
        # compute n-step lambda estimates
        def compute_nstep_lambda(carry, transition):
            lambda_return, truncated, importance_weight = carry
            # combine importance_weights with TD lambda
            done = transition.done
            reward = transition.soft_reward
            value = transition.value
            lambda_sum = (
                jnp.exp(importance_weight) * cfg.lmbda * lambda_return
                + (1 - jnp.exp(importance_weight) * cfg.lmbda) * value
            )
            delta = cfg.gamma * jnp.where(truncated, value, (1.0 - done) * lambda_sum)
            lambda_return = reward + delta
            truncated = transition.truncated
            return (
                lambda_return,
                truncated,
                transition.importance_weight,
            ), lambda_return

        _, target_values = maybe_lax_scan(
            compute_nstep_lambda,
            (
                batch.value[-1],
                jnp.ones_like(batch.truncated[0]),
                jnp.zeros_like(batch.importance_weight[0]),
            ),
            batch,
            reverse=True,
            use_scan=cfg.use_lax_scan,
        )
        # Reshape data to (num_steps * num_envs, ...)
        data = (batch, target_values)
        data = jax.tree.map(
            lambda x: x.reshape((cfg.num_steps * cfg.num_envs, *x.shape[2:])), data
        )

        train_state = train_state.replace(
            actor_target=train_state.actor_target.replace(
                params=train_state.actor.params
            ),
        )
        actor_target_model = nnx.merge(
            train_state.actor_target.graphdef, train_state.actor_target.params
        )

        def update(train_state, key) -> tuple[SACTrainState, dict[str, jax.Array]]:
            def minibatch_update(carry, indices):
                idx, train_state = carry
                # Sample data at indices from the batch
                minibatch, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )

                def critic_loss_fn(params):
                    critic_model = nnx.merge(train_state.critic.graphdef, params)
                    critic_pred = critic_model.critic_cat(
                        minibatch.critic_obs, minibatch.action
                    ).squeeze()
                    if cfg.hl_gauss:
                        target_cat = jax.vmap(
                            utils.hl_gauss, in_axes=(0, None, None, None)
                        )(target_values, cfg.num_bins, cfg.vmin, cfg.vmax)
                        critic_update_loss = optax.softmax_cross_entropy(
                            critic_pred, target_cat
                        )
                    else:
                        critic_update_loss = optax.squared_error(
                            critic_pred.reshape(-1,1),
                            target_values.reshape(-1,1),
                        )

                    # Aux loss
                    _, pred, pred_rew, value = critic_model.forward(
                        minibatch.critic_obs, minibatch.action
                    )
                    aux_loss = optax.squared_error(pred,  minibatch.next_emb)
                    aux_rew_loss = optax.squared_error(pred_rew, minibatch.reward.reshape(-1, 1))
                    aux_loss = jnp.mean(
                        (1 - minibatch.done.reshape(-1, 1))
                        * jnp.concatenate(
                            [aux_loss, aux_rew_loss], axis=-1
                        ), axis=-1)

                    # compute l2 error for logging
                    critic_loss = optax.squared_error(
                        value,
                        target_values,
                    )
                    critic_loss = jnp.mean(critic_loss)
                    loss = jnp.mean(
                        (1.0 - minibatch.truncated)
                        * (critic_update_loss + cfg.aux_loss_mult * aux_loss)
                    )
                    # log critic parameters norm
                    critic_pnorm = utils.tree_norm(params)

                    ## print the critic loss
                   # jax.debug.print("critic_loss: {cl}, aux_rew_loss: {arl}", cl=critic_loss, arl=aux_rew_loss.mean())
                    #jax.debug.print("minibatch.reward: {cl}, mean: {mean}, scaled_mean: {scaled_mean}",  cl=minibatch.reward.shape, mean=minibatch.reward.mean(), scaled_mean = minibatch.reward.mean()*cfg.diffusion.diff_steps)
                    return loss, dict(
                        value_loss=critic_loss,
                        critic_update_loss=critic_update_loss,
                        loss=loss,
                        aux_loss=aux_loss,
                        rew_aux_loss= aux_rew_loss,
                        q=value.mean(),
                        reward_mean=minibatch.reward.mean(),
                        target_values=target_values.mean(),
                        critic_pnorm=critic_pnorm,
                    )

                def actor_loss(params):
                    critic_target_model = nnx.merge(
                        train_state.critic.graphdef,
                        train_state.critic.params,
                    )
                    actor_model = nnx.merge(train_state.actor.graphdef, params)

                    # SAC actor loss
                    #key, subkey = jax.random.split(key)
                    ### TODO chekc what happens with the key here
                    pred_action, gen_log_prob, dest_log_prob = actor_model.vmap_sample_next_step(minibatch.obs, key)
                    entropy_prior = actor_model.get_prior_entropy()

                    log_prob_ratio = gen_log_prob - dest_log_prob

                    value = critic_target_model.critic(
                        minibatch.critic_obs, pred_action
                    )

                    entropy =  -cfg.diffusion.diff_steps*jnp.mean(log_prob_ratio, axis = 0) #+ entropy_prior
                    entropy = jax.lax.stop_gradient(entropy)
                    # log_w = actor_model.sample_complete_loop(minibatch.obs, key)
                    # new_entropy = - jnp.mean(log_w)
                    # jax.debug.print("entropy: {e}, entropy_prior: {ps}, dime_entropy: {de}, diff_steps: {ds}",
                    #                  e=entropy.mean(), ps=entropy_prior, de=-cfg.diffusion.diff_steps*jnp.mean(log_prob_ratio, axis = 0), ds=cfg.diffusion.diff_steps)


                    # policy KL constraint
                    if cfg.reverse_kl:
                        keys = jax.random.split(key, cfg.kl_action_rep)
                        def compute_kl_single(k):
                            return actor_model.rkl_div_one_step(k, minibatch.obs, actor_target_model, stop_grad=False)
                        
                        kl_log_ratios = jax.vmap(compute_kl_single)(keys)  # (kl_action_rep, batch_size, 1)
                        kl_log_ratios = kl_log_ratios.mean(axis=0)  # Average over samples => (batch_size, 1)

                        kl = cfg.diffusion.diff_steps*kl_log_ratios.sum(-1)
                    else:
                        keys = jax.random.split(key, cfg.kl_action_rep)
                        def compute_kl_single(k):
                            return actor_model.fkl_div_one_step(k, minibatch.obs, actor_target_model, stop_grad=False)
                        
                        kl_log_ratios = jax.vmap(compute_kl_single)(keys)  # (kl_action_rep, batch_size, 1)
                        kl_log_ratios = kl_log_ratios.mean(axis=0)  # Average over samples => (batch_size, 1)

                        kl = cfg.diffusion.diff_steps*kl_log_ratios.sum(-1)

                    lagrangian = actor_model.lagrangian()

                    # print log prob ratio and kl shape
                    #jax.debug.print("log_prob_ratio: {lpr}, kl: {k}", lpr=log_prob_ratio.shape, k=kl.shape)
                    #jax.debug.print("log_prob_ratio: {lpr}, kl: {k}", lpr=log_prob_ratio, k=kl)
                    if cfg.actor_kl_clip_mode == "full":
                        actor_loss = (
                            log_prob_ratio * jax.lax.stop_gradient(actor_model.temperature())
                            - value
                            + kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl
                        )
                    elif cfg.actor_kl_clip_mode == "clipped":
                        actor_loss = jnp.where(
                            kl < cfg.kl_bound,
                            log_prob_ratio * jax.lax.stop_gradient(actor_model.temperature())
                            - value,
                            kl * jax.lax.stop_gradient(lagrangian) * cfg.reduce_kl,
                        )
                    elif cfg.actor_kl_clip_mode == "value":
                        actor_loss = (
                            log_prob_ratio * jax.lax.stop_gradient(actor_model.temperature())
                            - value
                        )
                    else:
                        raise ValueError(
                            f"Unknown actor loss mode: {cfg.actor_kl_clip_mode}"
                        )

                    # SAC target entropy loss
                    target_entropy = action_size_target + entropy
                    target_entropy_loss = (
                        actor_model.temperature()
                        * jax.lax.stop_gradient(target_entropy)
                    )
                    target_entropy_loss = target_entropy_loss.mean()

                    # Lagrangian constraint (follows temperature update)
                    lagrangian_loss = -lagrangian * jax.lax.stop_gradient(
                        kl - cfg.kl_bound
                    )
                    lagrangian_loss = lagrangian_loss.mean()

                    # total loss
                    loss = jnp.mean(actor_loss)
                    if cfg.update_entropy_lagrangian:
                        loss += target_entropy_loss
                    if cfg.update_kl_lagrangian:
                        loss += lagrangian_loss

                    # log actor parameters norm
                    actor_pnorm = utils.tree_norm(params)

                    # log diffusion coefficient (detached for safe logging)
                    friction = actor_model.diffusion_model.friction.value
                    friction_detached = jax.lax.stop_gradient(friction)

                    # print with jax debug the entropy and kl values,a ctor and critic loss

                    #jax.debug.print("entropy: {e}, kl: {k}, actor_loss: {al}", e=entropy, k=kl.mean(), al=actor_loss.mean())

                    return loss, dict(
                        actor_loss=actor_loss,
                        loss=loss,
                        temp=actor_model.temperature(),
                        abs_batch_action=jnp.abs(minibatch.action).mean(),
                        abs_pred_action=jnp.abs(pred_action).mean(),
                        reward_mean=minibatch.reward.mean()*cfg.diffusion.diff_steps,
                        kl=kl.mean(),
                        lagrangian=lagrangian,
                        lagrangian_loss=lagrangian_loss,
                        run_cost=0.,
                        sto_cost=0.,
                        terminal_cost=0.,
                        entropy=entropy,
                        entropy_loss=target_entropy_loss,
                        target_values=target_values.mean(),
                        actor_pnorm=actor_pnorm,
                        friction=friction_detached.mean()
                    )

                critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
                output, critic_grads = critic_grad_fn(train_state.critic.params)
                #jax.debug.print("critic loss: {loss}", loss=output[0])
                critic_train_state = train_state.critic.apply_gradients(critic_grads)
                train_state = train_state.replace(
                    critic=critic_train_state,
                )
                critic_metrics = output[1]
                # log critic parameters norm
                critic_gnorm = utils.tree_norm(critic_grads)
                critic_metrics["critic_gnorm"] = critic_gnorm

                actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
                output, actor_grads = actor_grad_fn(train_state.actor.params)
                #jax.debug.print("actor loss: {loss}", loss=output[0])
                actor_train_state = train_state.actor.apply_gradients(actor_grads)
                train_state = train_state.replace(
                    actor=actor_train_state,
                )
                actor_metrics = output[1]
                # log actor parameters norm
                actor_gnorm = utils.tree_norm(actor_grads)
                actor_metrics["actor_gnorm"] = actor_gnorm
                return (idx + 1, train_state), {
                    **critic_metrics,
                    **actor_metrics,
                }

            # Shuffle data and split into mini-batches
            key, shuffle_key = jax.random.split(key)
            mini_batch_size = (cfg.num_steps * cfg.num_envs) // cfg.num_mini_batches
            indices = jax.random.permutation(shuffle_key, cfg.num_steps * cfg.num_envs)
            minibatch_idxs = jax.tree.map(
                lambda x: x.reshape(
                    (cfg.num_mini_batches, mini_batch_size, *x.shape[1:])
                ),
                indices,
            )

            #jax.debug.print("Learning step {step}", step=train_state.iteration)
            jax.debug.print("n_minibatch {n}, mini_batch_size {size}", n=cfg.num_mini_batches, size=mini_batch_size)
            # Run model update for each mini-batch
            train_state, metrics = maybe_lax_scan(
                minibatch_update,
                train_state,
                minibatch_idxs,
                use_scan=cfg.use_lax_scan,
            )
            # Compute mean metrics across mini-batches
            metrics = jax.tree.map(lambda x: x.mean(0), metrics)
            return train_state, metrics

        # Update the model for a number of epochs
        key, train_key = jax.random.split(key)
        (_, train_state), update_metrics = maybe_lax_scan(
            f=update,
            init=(1, train_state),
            xs=jax.random.split(train_key, cfg.num_epochs),
            use_scan=cfg.use_lax_scan,
        )
        # Get metrics from the last epoch
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)

        return train_state, update_metrics

    def train_fn(key: PRNGKey, cfg: ReppoConfig) -> tuple[SACTrainState, dict]:
        def train_eval_step(key, train_state):
            def train_step(
                state: SACTrainState, key: PRNGKey
            ) -> tuple[SACTrainState, tuple[dict[str, jax.Array], Transition]]:
                key, rollout_key, learn_key = jax.random.split(key, 3)
                transitions, state = collect_rollout(key=rollout_key, train_state=state)
                state, update_metrics = learn_step(
                    key=learn_key, train_state=state, batch=transitions
                )
                metrics = {**update_metrics}
                state = state.replace(iteration=state.iteration + 1)
                return state, metrics

            train_key, eval_key = jax.random.split(key)
            eval_interval = int(
                (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
            )
            train_state, train_metrics = maybe_lax_scan(
                f=train_step,
                init=train_state,
                xs=jax.random.split(train_key, eval_interval),
                use_scan=cfg.use_lax_scan,
            )
            train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)
            
            # Get normalization state if needed
            if cfg.normalize_env:
                norm_state = train_state.last_env_state
            else:
                norm_state = None
            
            # Split keys for each evaluation - use same init seed for all ODE coefs
            eval_key, init_seed_key = jax.random.split(eval_key)
            
            # Evaluate with SDE (stochastic) - default
            eval_metrics = sde_eval_fn(init_seed_key, train_state, norm_state)
            
            # Evaluate with different ODE coefficients if specified
            if getattr(cfg, "ode_coefs", None) is not None and len(cfg.ode_coefs) > 0:
                for ode_coef in cfg.ode_coefs:
                    eval_metrics_ode = ode_eval_fn(
                        init_seed_key, train_state, ode_coef, norm_state
                    )
                    
                    ode_suffix = f"ode_{int(ode_coef * 100):03d}"
                    eval_metrics.update({f"{k}_{ode_suffix}": v for k, v in eval_metrics_ode.items()})
            
            train_returns = {
                "train/episode_return": train_state.last_env_state.info[
                    "returned_episode_returns"
                ].mean(),
                "train/episode_length": train_state.last_env_state.info[
                    "returned_episode_lengths"
                ].mean(),
            }

            metrics = {
                "time_step": train_state.time_steps,
                **utils.prefix_dict("train", train_metrics),
                **utils.prefix_dict("eval", eval_metrics),
                **train_returns,
            }
            return train_state, metrics

        def loop_body(
            train_state: SACTrainState, key: PRNGKey
        ) -> tuple[SACTrainState, dict]:
            key, subkey = jax.random.split(key)
            train_state, metrics = jax.vmap(train_eval_step)(
                jax.random.split(subkey, num_seeds), train_state
            )

            jax.debug.callback(log_callback, train_state, metrics)
            return train_state, metrics

        eval_interval = int(
            (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
        )
        num_train_steps = cfg.total_time_steps // (cfg.num_steps * cfg.num_envs)
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, init_key = jax.random.split(key)
        train_state = jax.vmap(make_init(cfg, env, env_params))(
            jax.random.split(init_key, num_seeds)
        )
        keys = jax.random.split(key, num_iterations)

        jax.debug.print("Starting training for {n} iterations", n=num_iterations)
        jax.debug.print("num_train_steps {n}", n=num_train_steps)

        state, metrics = maybe_lax_scan(
            f=loop_body,
            init=train_state,
            xs=keys,
            use_scan=cfg.use_lax_scan,
        )
        return state, metrics

    return train_fn

# type object
def _get_optuna_type(trial: optuna.Trial, name, values: list):
    if all(isinstance(v, int) for v in values):
        return trial.suggest_int(name, low=min(values), high=max(values))
    elif all(isinstance(v, float) for v in values):
        return trial.suggest_float(name, low=min(values), high=max(values))
    elif all(isinstance(v, str) for v in values):
        return trial.suggest_categorical(name, values)
    elif all(isinstance(v, bool) for v in values):
        return trial.suggest_categorical(name, [True, False])
    else:
        raise ValueError("Values must be of the same type (int, float, or str).")

def run(cfg: DictConfig, trial: optuna.Trial | None) -> float:
    """
    Run a single trial of the SAC training process with hyperparameter tuning.
    Args:
        cfg (DictConfig): Configuration for the SAC training.
        trial (optuna.Trial | None): Optuna trial object for hyperparameter tuning.
    Returns:
        float: The mean episode return from the trial.
    """
    sweep_metrics = []

    if trial is not None:
        # Set hyperparameters from the trial
        for name, values in cfg.trial_spec.items():
            if name in cfg.hyperparameters:
                sampled_value = _get_optuna_type(trial, name, values)
                # TODO: Why the fuck is this happening
                if isinstance(sampled_value, np.float64):
                    sampled_value = float(sampled_value)
                cfg.hyperparameters[name] = sampled_value
            else:
                raise ValueError(f"Hyperparameter {name} not found in config.")

    try:
        with open("completed_trials.txt", "r") as f:
            completed_trials = int(f.read())
    except FileNotFoundError:
        completed_trials = 0

    metric_history = []

    def log_callback(state, metrics):
        metrics["sys_time"] = time.perf_counter()
        if len(metric_history) > 0:
            num_env_steps = state.time_steps[0] - metric_history[-1]["time_step"][0]
            seconds = metrics["sys_time"] - metric_history[-1]["sys_time"]
            sps = num_env_steps / seconds
        else:
            sps = 0

        metric_history.append(metrics)
        episode_return = metrics["eval/episode_return"].mean()
        eval_length = metrics["eval/episode_length"].mean()

        lr_cfg = cfg.hyperparameters
        actor_step = int(np.asarray(state.actor.step).max())
        if lr_cfg.anneal_lr:
            num_iterations = (
                lr_cfg.total_time_steps // lr_cfg.num_steps // lr_cfg.num_envs
            )
            num_updates = max(
                1, num_iterations * lr_cfg.num_epochs * lr_cfg.num_mini_batches
            )
            progress = min(actor_step / num_updates, 1.0)
            current_lr = float((1.0 - progress) * lr_cfg.lr)
        else:
            current_lr = float(lr_cfg.lr)
        
        log_msg = f"step={state.time_steps[0]} episode_return={episode_return:.3f}, episode_length={eval_length:.3f}, lr={current_lr:.6f}"
        
        # Log ODE metrics if available
        ode_metrics = {}
        for key in metrics.keys():
            if "episode_return_ode_" in key:
                # Extract ODE coefficient from key (e.g., "eval/episode_return_ode_050" -> 0.50)
                ode_coef_str = key.split("_ode_")[-1]
                ode_coef = float(ode_coef_str) / 100.0
                ode_return = metrics[key].mean()
                ode_metrics[f"ode_{ode_coef}"] = ode_return
                log_msg += f", ode_{ode_coef}_return={ode_return:.3f}"
        
        log_msg += f" sps={sps:.2f}"
        logging.info(log_msg)
        
        log_data = {
            "eval/episode_return": episode_return,
            "eval/episode_length": eval_length,
            "train/lr": current_lr,
            # performance metric: steps per second
            "sps": sps,
            **jax.tree.map(jnp.mean, utils.filter_prefix("train", metrics)),
        }
        
        # Add all eval metrics (SDE and ODE variants)
        for key, value in metrics.items():
            if key.startswith("eval/"):
                log_data[key] = value.mean() if hasattr(value, 'mean') else value

        wandb.log(log_data, step=state.time_steps[0])

    # Set up the experiment
    if cfg.env.type == "brax":
        raise ValueError("Wrappers are not implemented yet")
        env = BraxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            reward_scaling=cfg.env.reward_scaling,
            terminate=cfg.env.terminate,
        )
    elif cfg.env.type == "mjx":
        env = MjxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            reward_scale=cfg.env.reward_scaling,
            push_distractions=cfg.env.get("push_distractions", False),
            asymmetric_observation=cfg.env.get("asymmetric_observation", False),
        )
        diff_cfg = cfg.hyperparameters.diffusion
        env_action_clip_value = cfg.hyperparameters.env_action_clip_value
        env = MjxDiffEnvWrapper(env, num_diff_steps = diff_cfg.diff_steps, diffusion_config=diff_cfg, low= -env_action_clip_value, high=env_action_clip_value)
    else:
        raise ValueError(f"Unknown environment type: {cfg.env.type}")

    train_fn = make_train_fn(
        cfg=ReppoConfig(**cfg.hyperparameters),
        env=env,
        log_callback=log_callback,
        num_seeds=cfg.num_seeds,
        reward_scale=1.0 / cfg.env.reward_scaling,
    )

    for i in range(completed_trials, cfg.num_trials):
        cfg.seed = cfg.seed + i

        wandb.init(
            mode=cfg.wandb.mode,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=[
                cfg.name,
                cfg.env.name,
                cfg.env.type,
                "hp_tune" if trial is not None else "val",
                *cfg.tags,
            ],
            config=OmegaConf.to_container(cfg),
            name=f"{cfg.name}-{cfg.env.name.lower()}",
            save_code=True,
        )

        logging.info(OmegaConf.to_yaml(cfg))

        key = jax.random.PRNGKey(cfg.seed)
        start = time.perf_counter()
        _, metrics = jax.jit(train_fn, static_argnums=(1,))(
            key, ReppoConfig(**cfg.hyperparameters)
        )
        jax.block_until_ready(metrics)
        duration = time.perf_counter() - start

        # Save metrics and finish the run
        logging.info(f"Training took {duration:.2f} seconds.")
        jnp.savez("metrics.npz", **metrics)
        wandb.finish()

        sweep_metrics.append(metrics["eval/episode_return"])

        with open("completed_trials.txt", "w") as f:
            f.write(str(i))

    sweep_metrics_array = jnp.array(sweep_metrics)
    return (0.1 * sweep_metrics_array.mean() + sweep_metrics_array[:, -1].mean()).item()


@hydra.main(version_base=None, config_path="../../config", config_name="reppo_dmerl")
def main(cfg: DictConfig):
    cfg.hyperparameters = OmegaConf.merge(cfg.hyperparameters, cfg.experiment_overrides.hyperparameters)
    run(cfg, trial=None)


if __name__ == "__main__":
    main()
