from functools import partial
from typing import Any, Tuple, Union
from src.networks.diffusion.utils import inverse_softplus
import chex
import gymnax
import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from flax import struct
from gymnax.environments import environment, spaces
from gymnax.environments.environment import Environment
from gymnax.environments.spaces import Box
from ml_collections import ConfigDict
from mujoco_playground import MjxEnv, registry
from src.env_utils.planar_path_env import PlanarPathEnv
from src.env_utils.turning_double_well_env import TurningDoubleWellEnv
from mujoco_playground._src.wrapper import wrap_for_brax_training, Wrapper
import distrax


class MjxGymnaxWrapper(Environment):
    def __init__(
        self,
        env_or_name: str | MjxEnv,
        episode_length: int = 1000,
        action_repeat: int = 1,
        reward_scale: float = 1.0,
        push_distractions: bool = False,
        config: dict = None,
        asymmetric_observation: bool = False,
    ):
        if isinstance(env_or_name, str):
            if env_or_name == "PlanarPathEnv":
                self.env = PlanarPathEnv()
                self.sanitize_nans = False
                self.reward_scale = reward_scale
                self.episode_length = episode_length
                if isinstance(self.env.observation_size, int):
                    self.dict_obs = False
                else:
                    self.dict_obs = True
                if asymmetric_observation:
                    self.dict_obs_key = "privileged_state"
                else:
                    self.dict_obs_key = "state"
                print(self.dict_obs_key)
                super().__init__()
                return
            if env_or_name == "TurningDoubleWellEnv":
                env_kwargs = dict(config) if config is not None else {}
                env_kwargs.setdefault("horizon", episode_length or 200)
                self.env = TurningDoubleWellEnv(**env_kwargs)
                self.sanitize_nans = False
                self.reward_scale = reward_scale
                self.episode_length = self.env.horizon
                if isinstance(self.env.observation_size, int):
                    self.dict_obs = False
                else:
                    self.dict_obs = True
                if asymmetric_observation:
                    self.dict_obs_key = "privileged_state"
                else:
                    self.dict_obs_key = "state"
                print(self.dict_obs_key)
                super().__init__()
                return
            if config is None:
                config = registry.get_default_config(env_or_name)
                is_humanoid_task = env_or_name in [
                    "G1JoystickRoughTerrain",
                    "G1JoystickFlatTerrain",
                    "T1JoystickRoughTerrain",
                    "T1JoystickFlatTerrain",
                ]
                if is_humanoid_task:
                    config.push_config.enable = push_distractions
            else:
                config = ConfigDict(config)
            env = registry.load(env_or_name, config=config)
            if episode_length is not None:
                env = wrap_for_brax_training(
                    env, episode_length=episode_length, action_repeat=action_repeat
                )
            self.env = env
            self.sanitize_nans = "humanoid" in env_or_name.lower()
        else:
            self.env = env_or_name
            self.sanitize_nans = False
        self.reward_scale = reward_scale
        self.episode_length = episode_length
        if isinstance(self.env.observation_size, int):
            self.dict_obs = False
        else:
            self.dict_obs = True

        if asymmetric_observation:
            self.dict_obs_key = "privileged_state"
        else:
            self.dict_obs_key = "state"
        print(self.dict_obs_key)
        super().__init__()

    def action_space(self, params):
        return gymnax.environments.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_size,),
        )

    def observation_space(self, params):
        def _with_nan_token(box: Box) -> Box:
            if not self.sanitize_nans:
                return box
            shape = box.shape
            if isinstance(shape, int):
                shape = (shape,)
            new_shape = shape[:-1] + (shape[-1] + 1,)
            return Box(low=box.low, high=box.high, shape=new_shape)

        if self.dict_obs:
            return _with_nan_token(Box(
                low=-float("inf"),
                high=float("inf"),
                shape=self.env.observation_size["state"],
            )), _with_nan_token(Box(
                low=-float("inf"),
                high=float("inf"),
                shape=self.env.observation_size[self.dict_obs_key],
            ))
        else:
            return _with_nan_token(Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.env.observation_size,),
            )), _with_nan_token(Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.env.observation_size,),
            ))

    @property
    def default_params(self) -> gymnax.EnvParams:
        return gymnax.EnvParams()

    def reset(self, key):
        state = self.env.reset(key)
        # state.info["truncation"] = 0.0
        obs = state.obs if not self.dict_obs else state.obs["state"]
        critic_obs = state.obs if not self.dict_obs else state.obs[self.dict_obs_key]
        if self.sanitize_nans:
            nan_token = self._nan_token(obs, jnp.array(False))
            obs = jnp.concatenate([obs, nan_token], axis=-1)
            critic_nan_token = self._nan_token(critic_obs, jnp.array(False))
            critic_obs = jnp.concatenate([critic_obs, critic_nan_token], axis=-1)
        return obs, critic_obs, state

    def step(self, key, state, action):
        # action = jnp.nan_to_num(action, 0.0)
        prev_state = state
        state = self.env.step(state, action)
        obs = state.obs if not self.dict_obs else state.obs["state"]
        critic_obs = state.obs if not self.dict_obs else state.obs[self.dict_obs_key]
        if self.sanitize_nans:
            has_nan = ~jnp.isfinite(obs).all()
            has_nan = jnp.logical_or(has_nan, ~jnp.isfinite(critic_obs).all())
            has_nan = jnp.logical_or(has_nan, ~jnp.isfinite(state.reward))
            prev_obs = (
                prev_state.obs if not self.dict_obs else prev_state.obs["state"]
            )
            prev_critic_obs = (
                prev_state.obs
                if not self.dict_obs
                else prev_state.obs[self.dict_obs_key]
            )
            obs = jnp.where(jnp.isfinite(obs), obs, prev_obs)
            critic_obs = jnp.where(
                jnp.isfinite(critic_obs), critic_obs, prev_critic_obs
            )
            min_reward = jnp.finfo(state.reward.dtype).min
            max_reward = jnp.finfo(state.reward.dtype).max
            avrg_reward = min_reward / 2 + max_reward / 2
            reward = jnp.where(jnp.isfinite(state.reward), state.reward, avrg_reward)
            done = (state.done > 0.5) | has_nan
            nan_token = self._nan_token(obs, has_nan)
            obs = jnp.concatenate([obs, nan_token], axis=-1)
            critic_nan_token = self._nan_token(critic_obs, has_nan)
            critic_obs = jnp.concatenate([critic_obs, critic_nan_token], axis=-1)
            return (
                obs,
                critic_obs,
                state,
                reward * self.reward_scale,
                done,
                {},
            )
        #print the step of the current state also print truncation
        # jax.debug.print("Orignial Step truncation={}", state.info["truncation"])
        # jax.debug.print("Orignial Env step info={}", state.info["steps"])
        # jax.debug.print("Env step reward={}", state.reward)
        # jax.debug.print("Orignial Env state.done={}", state.done)
        return (
            obs,
            critic_obs,
            state,
            state.reward * self.reward_scale,
            state.done > 0.5,
            {},
        )

    def _nan_token(self, obs, has_nan):
        token = jnp.asarray(has_nan, dtype=obs.dtype)
        if token.shape == ():
            token = jnp.broadcast_to(token, obs.shape[:-1])
        token = jnp.expand_dims(token, axis=-1)
        return token


@struct.dataclass
class MjxDiffEnvState:
    env_state: Any
    obs: jnp.ndarray
    critic_obs: jnp.ndarray
    done: jnp.ndarray
    truncated: jnp.ndarray
    diff_time_step: jnp.ndarray
    steps_since_reset: jnp.ndarray
    info: Any = None

    def unwrapped(self):
        return self.env_state

    def set_env_state(self, env_state):
        return self.replace(env_state=env_state)


def build_obs_dict(obs, orig_actions, diff_time_step):
    return {
        "orig_obs": obs.copy(),
        "orig_actions": orig_actions.copy(),
        "normed_actions": orig_actions.copy(),
        "diff_time_step": diff_time_step.copy(),
    }

class MjxDiffEnvWrapper(Wrapper):
    """Wraps MJX envs with diffusion metadata and long-horizon resets."""

    def __init__(self, env: MjxGymnaxWrapper, num_diff_steps: int, diffusion_config: ConfigDict, low = -0.999, high = 0.999):
        self.low = low
        self.high = high
        super().__init__(env)
        if num_diff_steps <= 0:
            raise ValueError("num_diff_steps must be positive.")
        if env.episode_length is None:
            raise ValueError("MjxDiffEnvWrapper requires env.episode_length to be set.")
        self.env = env
        self.num_diff_steps = num_diff_steps
        self.reset_after_steps = num_diff_steps * env.episode_length
        self._obs_space, self._critic_obs_space = self.env.observation_space(
            self.env.default_params
        )
        self._action_space = self.env.action_space(self.env.default_params)
        self.action_dim = self._action_space.shape[0]
        self.diffusion_config = diffusion_config
        self._init_prior_params()

    def _init_prior_params(self):
        init_std = self.diffusion_config.get("init_std", None)
        self.prior_mean = jnp.zeros((self.action_dim,))
        self.prior_std = jnp.ones((self.action_dim,)) * inverse_softplus(init_std)
        self.distribution = distrax.MultivariateNormalDiag(
            self.prior_mean, jax.nn.softplus(self.prior_std)
        )

    def prior_sampler(self, key, n_samples = 1):
        """Sample from the prior distribution.
        
        Args:
            key: JAX random key
            n_samples: Number of samples to generate (batch size)
            
        Returns:
            Samples of shape (n_samples, action_dim)
        """
        # Ensure n_samples is a Python int for sample_shape
        key, subkey = jax.random.split(key) 
        if isinstance(n_samples, jax.Array):
            n_samples = int(n_samples)
        
        samples = self.distribution.sample(seed=subkey)
        return samples, key
    
    def vmap_prior_samples(self, key, n_samples = 1):
        """Vectorized prior sampler over batch dimension.
        
        Args:
            key: JAX random key of shape (batch_size, 2)
            n_samples: Number of samples to generate per batch element
            
        Returns:
            Samples of shape (batch_size, n_samples, action_dim)
        """
        in_axes = (0, None)  # key has batch dimension, n_samples is static
        out = jax.vmap(self.prior_sampler, in_axes=in_axes)(key, n_samples)
        return out  # shape: (batch_size, n_samples, action_dim)

    def _zero_action_array(self, obs_like):
        zeros_shape = obs_like.shape[:-1] + self._action_space.shape
        return jnp.zeros(zeros_shape, dtype=jnp.float32)

    def _zero_time_array(self, obs_like):
        return jnp.zeros(obs_like.shape[:-1] + (1,), dtype=jnp.float32)

    def _build_obs(self, obs, orig_actions, diff_time_step):
        return build_obs_dict(obs, orig_actions, diff_time_step)

    def observation_space(self, params=None):
        actor_space = {
            "orig_obs": self._obs_space,
            "orig_actions": spaces.Box(
                low=-jnp.inf, high=jnp.inf, shape=self._action_space.shape
            ),
            "normed_actions": spaces.Box(
                low=-1, high=1, shape=self._action_space.shape
            ),
            "diff_time_step": spaces.Box(low=0.0, high=self.num_diff_steps, shape=(1,)),
        }
        critic_space = {
            "orig_obs": self._critic_obs_space,
            "orig_actions": actor_space["orig_actions"],
            "normed_actions": actor_space["normed_actions"],
            "diff_time_step": actor_space["diff_time_step"],
        }
        if hasattr(spaces, "Dict"):
            return spaces.Dict(actor_space), spaces.Dict(critic_space)
        return actor_space, critic_space
    
    def get_obs_space_sizes(self, params=None):
        actor_space, critic_space = self.observation_space(params)

        def _extract(space, key):
            if hasattr(space, "spaces"):   # gymnax Dict
                return space.spaces[key]
            return space[key]

        actor_orig = _extract(actor_space, "orig_obs")
        actor_tanh = _extract(actor_space, "normed_actions")
        critic_orig = _extract(critic_space, "orig_obs")
        critic_tanh = _extract(critic_space, "normed_actions")

        obs_dim = actor_orig.shape[0] + actor_tanh.shape[0]
        critic_dim = critic_orig.shape[0] + critic_tanh.shape[0] 
        return obs_dim, critic_dim

    def action_space(self, params=None):
        return self._action_space

    def reset(self, key, params=None):
        obs, critic_obs, env_state = self.env.reset(key)
        prior_actions, key = self.reset_actions(key, obs.shape[0])
        diff_time_step = self._zero_time_array(obs)
        obs_dict = self._build_obs(obs, prior_actions, diff_time_step)
        critic_obs_dict = self._build_obs(critic_obs, prior_actions, diff_time_step)
        zeros = jnp.zeros((obs.shape[0],), dtype=jnp.bool_)
        truncated = jnp.zeros((obs.shape[0],), dtype=jnp.float32)
        base_info = dict(getattr(env_state, "info", {}))
        base_info["truncation"] = truncated

        state = MjxDiffEnvState(
            env_state=env_state,
            obs=obs,
            critic_obs=critic_obs,
            done=zeros,
            truncated=truncated,
            info=base_info,
            diff_time_step=diff_time_step,
            steps_since_reset=jnp.zeros_like(diff_time_step),
        )
        return obs_dict, critic_obs_dict, state
    
    def reset_actions(self, key, n_envs):
        prior_actions, key = self.vmap_prior_samples(key)
        return prior_actions, key
    
    def reset_diff_time_steps(self, n_envs):
        return self._zero_time_array(jnp.zeros((n_envs, 1)))

    def orig_env_and_reset_actions_step(self, args):
        key, state, action, diff_time_steps, steps_since_reset = args
        scaled_action = jnp.tanh(action)
        scaled_action = jnp.clip(scaled_action, self.low, self.high)
        obs, critic_obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, scaled_action
        )
        truncated = env_state.info.get(
            "truncation", jnp.zeros((obs.shape[0],), dtype=jnp.float32)
        )
        info = dict(env_state.info)
        info["truncation"] = truncated
        # jax.debug.print("env reset? {d}", d=done)
        # reset_mask = env_state.info.get("returned_episode", None)
        # if reset_mask is not None:
        #     jax.debug.print("returned_episode mask: {m}", m=reset_mask)
        #print the reward with jax debug
        #jax.debug.print("Reset step reward: {r}", r=reward)
        prior_actions, key = self.reset_actions(key, obs.shape[0])
        diff_time_steps = self.reset_diff_time_steps(obs.shape[0])
        obs_dict = self._build_obs(obs, prior_actions, diff_time_steps)
        critic_obs_dict = self._build_obs(critic_obs, prior_actions, diff_time_steps)

        return (
            obs_dict,
            critic_obs_dict,
            env_state,
            reward,
            done,
            truncated,
            obs,
            critic_obs,
            diff_time_steps,
            steps_since_reset,
            info,
        )
    
    def diff_env_step(self, args):
        key, state, action, diff_time_steps, steps_since_reset = args
        obs_dict = self._build_obs(state.obs, action, diff_time_steps)
        critic_obs_dict = self._build_obs(state.critic_obs, action, diff_time_steps)
        reward = jnp.zeros((obs_dict["orig_obs"].shape[0],), dtype=jnp.float32)
        # set done to false
        done = jnp.zeros((obs_dict["orig_obs"].shape[0],), dtype=jnp.bool_)
        truncated = jnp.zeros((obs_dict["orig_obs"].shape[0],), dtype=jnp.float32)
        info = dict(state.info) if state.info is not None else {}
        info["truncation"] = truncated
        return (
            obs_dict,
            critic_obs_dict,
            state.env_state,
            reward,
            done,
            truncated,
            state.obs,
            state.critic_obs,
            diff_time_steps,
            steps_since_reset,
            info,
        )

    def step(self, key, state: MjxDiffEnvState, action):
        diff_time_step = state.diff_time_step + jnp.ones_like(state.diff_time_step)
        steps_since_reset = state.steps_since_reset + jnp.ones_like(state.steps_since_reset)
        reset_due = jnp.any(diff_time_step >= self.num_diff_steps)
        #print reset due with jax debug and also diff_time_step
        # jax.debug.print("Diff time step: {dts}, Reset due: {rd}", dts=diff_time_step, rd=reset_due)

        (   obs_dict,
            critic_obs_dict,
            env_state,
            reward,
            done,
            truncated,
            raw_obs,
            raw_critic_obs,
            new_diff_time,
            new_steps_since_reset,
            info,
        ) = jax.lax.cond(
            reset_due,
            self.orig_env_and_reset_actions_step,
            self.diff_env_step,
            operand=(key, state, action, diff_time_step, steps_since_reset),
        )
        #jax.debug.print("selected reward: {r}", r=reward)
        #jax.debug.print("Step info={}", info)
        # jax.debug.print("diff Env step info={}", state.info["steps"])
        # jax.debug.print("diff Env step reward={}", reward)
        # jax.debug.print("diff Env state.done={}", state.done)
        # jax.debug.print("diff Step truncation={}", state.info["truncation"])

        new_state = MjxDiffEnvState(
            env_state=env_state,
            obs=raw_obs,
            critic_obs=raw_critic_obs,
            done=done > 0.5,
            truncated=truncated,
            info=info,
            diff_time_step=new_diff_time,
            steps_since_reset=new_steps_since_reset,
        )
        return obs_dict, critic_obs_dict, new_state, reward, done, info



@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray
    timestep: jnp.ndarray
    truncated: jnp.ndarray
    info: Any = None

    def unwrapped(self):
        return self.env_state

    def set_env_state(self, env_state):
        return self.replace(env_state=env_state)


class LogWrapper(Wrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment, num_envs: int):
        super().__init__(env)
        self.num_envs = num_envs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key) -> Tuple[chex.Array, environment.EnvState]:
        obs, critic_obs, env_state = self.env.reset(key)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.zeros((self.num_envs,)),
            episode_lengths=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            returned_episode_returns=jnp.zeros((self.num_envs,)),
            returned_episode_lengths=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            timestep=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            truncated=jnp.ones((self.num_envs,), dtype=jnp.float32),
            info={
                "returned_episode": jnp.zeros((self.num_envs,), dtype=jnp.bool_),
                "returned_episode_returns": jnp.zeros((self.num_envs,)),
                "timestep": jnp.zeros((self.num_envs,), dtype=jnp.int32),
                "returned_episode_lengths": jnp.zeros(
                    (self.num_envs,), dtype=jnp.int32
                ),
            },
        )
        return obs, critic_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, critic_obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, action
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        info = {}
        info["returned_episode_returns"] = (
            state.returned_episode_returns * (1 - done) + new_episode_return * done
        )
        info["returned_episode_lengths"] = (
            state.returned_episode_lengths * (1 - done) + new_episode_length * done
        )
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        ##print episode_return and episone_lengths with jax.debug
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
            truncated=env_state.info["truncation"],
            info=info,
        )
        return obs, critic_obs, state, reward, done, info



class BraxGymnaxWrapper:
    def __init__(
        self,
        env_name,
        backend="generalized",
        episode_length=1000,
        reward_scaling=1.0,
        terminate=True,
    ):
        env = envs.get_environment(
            env_name=env_name, backend=backend, terminate_when_unhealthy=terminate
        )
        env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=1)
        env = AutoResetWrapper(env)
        self.env = env
        self.action_size = self.env.action_size
        self.observation_size = (self.env.observation_size,)
        self.default_params = ()
        self.reward_scaling = reward_scaling

    def reset(self, key):
        def _reset_single(k):
            state = self.env.reset(k)
            obs = state.obs
            return obs, obs, state

        if key.ndim == 1:
            return _reset_single(key)
        obs, critic_obs, state = jax.vmap(_reset_single)(key)
        return obs, critic_obs, state

    def step(self, key, state, action):
        next_state = self.env.step(state, action)
        return (
            next_state.obs,
            next_state.obs,
            next_state,
            next_state.reward * self.reward_scaling,
            next_state.done > 0.5,
            {},
        )

    def observation_space(self, params = None):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.env.observation_size,),
        ), spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.env.observation_size,),
        )

    def action_space(self, params = None):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_size,),
        )

class ClipAction(Wrapper):
    def __init__(self, env, low=-0.999, high=0.999):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action):
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self.env.step(key, state, action)
    
class TanhClipAction(Wrapper):
    def __init__(self, env, low=-0.999, high=0.999):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action):
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        #action = jnp.clip(action, self.low, self.high)
        return self.env.step(key, state, action)


@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    critic_mean: jnp.ndarray
    critic_var: jnp.ndarray
    reward_mean: jnp.ndarray
    reward_var: jnp.ndarray
    count: float
    env_state: environment.EnvState
    truncated: float
    info: Any = None

    def unwrapped(self):
        return self.env_state.unwrapped()

    def set_env_state(self, env_state):
        return self.replace(env_state=self.env_state.set_env_state(env_state))


class NormalizeVec(Wrapper):
    def __init__(self, env, normalize_reward: bool = False):
        super().__init__(env)
        self.normalize_reward = normalize_reward

    def _init_state(self, key):
        obs, critic_obs, env_state = self.env.reset(key)
        reward_mean = jnp.array(0.0, dtype=obs.dtype)
        reward_var = jnp.array(1.0, dtype=obs.dtype)
        return NormalizeVecObsEnvState(
            mean=jnp.mean(obs, axis=0),
            var=jnp.var(obs, axis=0),
            critic_mean=jnp.mean(critic_obs, axis=0),
            critic_var=jnp.var(critic_obs, axis=0),
            reward_mean=reward_mean,
            reward_var=reward_var,
            count=obs.shape[0],
            env_state=env_state,
        )

    def _compute_stats(self, mean, var, count, obs):
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count

        return new_mean, new_var

    def reset(self, key, params=None):
        obs, critic_obs, env_state = self.env.reset(key)
        if params is not None:
            mean = params.mean
            var = params.var
            critic_mean = params.critic_mean
            critic_var = params.critic_var
            reward_mean = params.reward_mean
            reward_var = params.reward_var
            count = params.count
        else:
            mean = jnp.mean(obs, axis=0)
            var = jnp.var(obs, axis=0)
            critic_mean = jnp.mean(critic_obs, axis=0)
            critic_var = jnp.var(critic_obs, axis=0)
            reward_mean = jnp.array(0.0, dtype=obs.dtype)
            reward_var = jnp.array(1.0, dtype=obs.dtype)
            count = obs.shape[0]
        state = NormalizeVecObsEnvState(
            mean=mean,
            var=var,
            critic_mean=critic_mean,
            critic_var=critic_var,
            reward_mean=reward_mean,
            reward_var=reward_var,
            count=count,
            env_state=env_state,
            truncated=env_state.truncated,
            info=env_state.info,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-2),
            (critic_obs - state.critic_mean) / jnp.sqrt(state.critic_var + 1e-2),
            state,
        )

    def step(self, key, state, action):
        obs, critic_obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, action
        )

        new_mean, new_var = self._compute_stats(state.mean, state.var, state.count, obs)
        new_critic_mean, new_critic_var = self._compute_stats(
            state.critic_mean, state.critic_var, state.count, critic_obs
        )
        new_reward_mean, new_reward_var = self._compute_stats(
            state.reward_mean, state.reward_var, state.count, reward
        )

        new_count = state.count + obs.shape[0]

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            critic_mean=new_critic_mean,
            critic_var=new_critic_var,
            reward_mean=new_reward_mean,
            reward_var=new_reward_var,
            count=new_count,
            env_state=env_state,
            truncated=env_state.truncated,
            info=env_state.info,
        )
        if self.normalize_reward:
            reward = (reward - state.reward_mean) / jnp.sqrt(state.reward_var + 1e-2)
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-2),
            (critic_obs - state.critic_mean) / jnp.sqrt(state.critic_var + 1e-2),
            state,
            reward,
            done,
            info,
        )


@struct.dataclass
class DiffNormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    action_mean: jnp.ndarray
    action_var: jnp.ndarray
    critic_mean: jnp.ndarray
    critic_var: jnp.ndarray
    critic_action_mean: jnp.ndarray
    critic_action_var: jnp.ndarray
    reward_mean: jnp.ndarray
    reward_var: jnp.ndarray
    count: float
    env_state: environment.EnvState
    truncated: float
    info: Any = None

    def unwrapped(self):
        return self.env_state.unwrapped()

    def set_env_state(self, env_state):
        return self.replace(env_state=self.env_state.set_env_state(env_state))


class DiffNormalizeVec(Wrapper):
    """Normalize only the `orig_obs` entry within dict observations."""

    def __init__(self, env, normalize_reward: bool = False, num_diff_steps: int | None = None):
        super().__init__(env)
        self.normalize_reward = normalize_reward
        if num_diff_steps is None:
            raise ValueError("num_diff_steps must be provided for DiffNormalizeVec.")
        self.num_diff_steps = num_diff_steps

    def _compute_stats(self, mean, var, count, obs):
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count

        return new_mean, new_var

    def _normalize_obs_dict(
        self,
        obs_dict,
        obs_mean,
        obs_var,
        action_mean,
        action_var,
    ):
        updated = dict(obs_dict)
        updated["orig_obs"] = (obs_dict["orig_obs"] - obs_mean) / jnp.sqrt(
            obs_var + 1e-2
        )
        if "normed_actions" in obs_dict:
            updated["normed_actions"] = (
                obs_dict["normed_actions"] - action_mean
            ) / jnp.sqrt(action_var + 1e-2)
        return updated

    def reset(self, key, params=None):
        obs, critic_obs, env_state = self.env.reset(key)
        orig_obs = obs["orig_obs"]
        critic_orig_obs = critic_obs["orig_obs"]
        actor_actions = obs.get("normed_actions", None)
        critic_actions = critic_obs.get("normed_actions", None)
        if params is not None:
            mean = params.mean
            var = params.var
            action_mean = params.action_mean
            action_var = params.action_var
            critic_mean = params.critic_mean
            critic_var = params.critic_var
            critic_action_mean = params.critic_action_mean
            critic_action_var = params.critic_action_var
            reward_mean = params.reward_mean
            reward_var = params.reward_var
            count = params.count
        else:
            mean = jnp.mean(orig_obs, axis=0)
            var = jnp.var(orig_obs, axis=0)
            if actor_actions is not None:
                action_mean = jnp.mean(actor_actions, axis=0)
                action_var = jnp.var(actor_actions, axis=0)
            else:
                action_mean = jnp.array(0.0)
                action_var = jnp.array(1.0)
            critic_mean = jnp.mean(critic_orig_obs, axis=0)
            critic_var = jnp.var(critic_orig_obs, axis=0)
            if critic_actions is not None:
                critic_action_mean = jnp.mean(critic_actions, axis=0)
                critic_action_var = jnp.var(critic_actions, axis=0)
            else:
                critic_action_mean = jnp.array(0.0)
                critic_action_var = jnp.array(1.0)
            reward_mean = jnp.array(0.0, dtype=orig_obs.dtype)
            reward_var = jnp.array(1.0, dtype=orig_obs.dtype)
            count = orig_obs.shape[0]
        state = DiffNormalizeVecObsEnvState(
            mean=mean,
            var=var,
            action_mean=action_mean,
            action_var=action_var,
            critic_mean=critic_mean,
            critic_var=critic_var,
            critic_action_mean=critic_action_mean,
            critic_action_var=critic_action_var,
            reward_mean=reward_mean,
            reward_var=reward_var,
            count=count,
            env_state=env_state,
            truncated=env_state.truncated,
            info=env_state.info,
        )
        return (
            self._normalize_obs_dict(
                obs, state.mean, state.var, state.action_mean, state.action_var
            ),
            self._normalize_obs_dict(
                critic_obs,
                state.critic_mean,
                state.critic_var,
                state.critic_action_mean,
                state.critic_action_var,
            ),
            state,
        )

    def step(self, key, state: DiffNormalizeVecObsEnvState, action):
        obs, critic_obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, action
        )
        orig_obs = obs["orig_obs"]
        critic_orig_obs = critic_obs["orig_obs"]
        actor_actions = obs.get("normed_actions", None)
        critic_actions = critic_obs.get("normed_actions", None)

        new_mean, new_var = self._compute_stats(
            state.mean, state.var, state.count, orig_obs
        )
        new_critic_mean, new_critic_var = self._compute_stats(
            state.critic_mean, state.critic_var, state.count, critic_orig_obs
        )
        if actor_actions is not None:
            new_action_mean, new_action_var = self._compute_stats(
                state.action_mean, state.action_var, state.count, actor_actions
            )
        else:
            new_action_mean, new_action_var = state.action_mean, state.action_var
        if critic_actions is not None:
            new_critic_action_mean, new_critic_action_var = self._compute_stats(
                state.critic_action_mean,
                state.critic_action_var,
                state.count,
                critic_actions,
            )
        else:
            new_critic_action_mean, new_critic_action_var = (
                state.critic_action_mean,
                state.critic_action_var,
            )
        reward_count = state.count / self.num_diff_steps
        new_reward_mean, new_reward_var = self._compute_stats(
            state.reward_mean, state.reward_var, reward_count, reward
        )
        new_count = state.count + orig_obs.shape[0]

        state = DiffNormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            action_mean=new_action_mean,
            action_var=new_action_var,
            critic_mean=new_critic_mean,
            critic_var=new_critic_var,
            critic_action_mean=new_critic_action_mean,
            critic_action_var=new_critic_action_var,
            reward_mean=new_reward_mean,
            reward_var=new_reward_var,
            count=new_count,
            env_state=env_state,
            truncated=env_state.truncated,
            info=env_state.info,
        )
        if self.normalize_reward:
            reward = (reward - state.reward_mean) / jnp.sqrt(state.reward_var + 1e-2)

        return (
            self._normalize_obs_dict(
                obs, state.mean, state.var, state.action_mean, state.action_var
            ),
            self._normalize_obs_dict(
                critic_obs,
                state.critic_mean,
                state.critic_var,
                state.critic_action_mean,
                state.critic_action_var,
            ),
            state,
            reward,
            done,
            info,
        )
