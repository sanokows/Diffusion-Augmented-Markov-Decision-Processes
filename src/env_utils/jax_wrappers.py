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
        else:
            self.env = env_or_name
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
        if self.dict_obs:
            return Box(
                low=-float("inf"),
                high=float("inf"),
                shape=self.env.observation_size["state"],
            ), Box(
                low=-float("inf"),
                high=float("inf"),
                shape=self.env.observation_size[self.dict_obs_key],
            )
        else:
            return Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.env.observation_size,),
            ), Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.env.observation_size,),
            )

    @property
    def default_params(self) -> gymnax.EnvParams:
        return gymnax.EnvParams()

    def reset(self, key):
        state = self.env.reset(key)
        # state.info["truncation"] = 0.0
        obs = state.obs if not self.dict_obs else state.obs["state"]
        critic_obs = state.obs if not self.dict_obs else state.obs[self.dict_obs_key]
        return obs, critic_obs, state

    def step(self, key, state, action):
        # action = jnp.nan_to_num(action, 0.0)
        state = self.env.step(state, action)
        obs = state.obs if not self.dict_obs else state.obs["state"]
        critic_obs = state.obs if not self.dict_obs else state.obs[self.dict_obs_key]
        return (
            obs,
            critic_obs,
            state,
            state.reward * self.reward_scale,
            state.done > 0.5,
            {},
        )


@struct.dataclass
class MjxDiffEnvState:
    env_state: Any
    obs: jnp.ndarray
    critic_obs: jnp.ndarray
    done: jnp.ndarray
    diff_time_step: jnp.ndarray
    steps_since_reset: jnp.ndarray
    info: Any = None

    def unwrapped(self):
        return self.env_state

    def set_env_state(self, env_state):
        return self.replace(env_state=env_state)


def build_obs_dict(obs, orig_actions, diff_time_step):
    return {
        "orig_obs": obs,
        "orig_actions": orig_actions,
        "normed_actions": jnp.tanh(orig_actions),
        "diff_time_step": diff_time_step,
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
        base_info = getattr(env_state, "info", {})

        state = MjxDiffEnvState(
            env_state=env_state,
            obs=obs,
            critic_obs=critic_obs,
            done=zeros,
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
            obs,
            critic_obs,
            diff_time_steps,
            steps_since_reset,
        )
    
    def diff_env_step(self, args):
        key, state, action, diff_time_steps, steps_since_reset = args
        obs_dict = self._build_obs(state.obs, action, diff_time_steps)
        critic_obs_dict = self._build_obs(state.critic_obs, action, diff_time_steps)
        reward = jnp.zeros((obs_dict["orig_obs"].shape[0],), dtype=jnp.float32)
        done = state.done
        return (
            obs_dict,
            critic_obs_dict,
            state.env_state,
            reward,
            done,
            state.obs,
            state.critic_obs,
            diff_time_steps,
            steps_since_reset,
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
            raw_obs,
            raw_critic_obs,
            new_diff_time,
            new_steps_since_reset,
        ) = jax.lax.cond(
            reset_due,
            self.orig_env_and_reset_actions_step,
            self.diff_env_step,
            operand=(key, state, action, diff_time_step, steps_since_reset),
        )
        #jax.debug.print("selected reward: {r}", r=reward)
        info = env_state.info
        new_state = MjxDiffEnvState(
            env_state=env_state,
            obs=raw_obs,
            critic_obs=raw_critic_obs,
            done=done,
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
        #action = jnp.clip(action, self.low, self.high)
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
    count: float
    env_state: environment.EnvState
    truncated: float
    info: Any = None

    def unwrapped(self):
        return self.env_state.unwrapped()

    def set_env_state(self, env_state):
        return self.replace(env_state=self.env_state.set_env_state(env_state))


class NormalizeVec(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def _init_state(self, key):
        obs, critic_obs, env_state = self.env.reset(key)
        return NormalizeVecObsEnvState(
            mean=jnp.mean(obs, axis=0),
            var=jnp.var(obs, axis=0),
            critic_mean=jnp.mean(critic_obs, axis=0),
            critic_var=jnp.var(critic_obs, axis=0),
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
            count = params.count
        else:
            mean = jnp.mean(obs, axis=0)
            var = jnp.var(obs, axis=0)
            critic_mean = jnp.mean(critic_obs, axis=0)
            critic_var = jnp.var(critic_obs, axis=0)
            count = obs.shape[0]
        state = NormalizeVecObsEnvState(
            mean=mean,
            var=var,
            critic_mean=critic_mean,
            critic_var=critic_var,
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

        new_count = state.count + obs.shape[0]

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            critic_mean=new_critic_mean,
            critic_var=new_critic_var,
            count=new_count,
            env_state=env_state,
            truncated=env_state.truncated,
            info=env_state.info,
        )
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
    critic_mean: jnp.ndarray
    critic_var: jnp.ndarray
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

    def __init__(self, env):
        super().__init__(env)

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

    def _replace_orig_obs(self, obs_dict, new_obs):
        updated = dict(obs_dict)
        updated["orig_obs"] = new_obs
        return updated

    def reset(self, key, params=None):
        obs, critic_obs, env_state = self.env.reset(key)
        orig_obs = obs["orig_obs"]
        critic_orig_obs = critic_obs["orig_obs"]
        if params is not None:
            mean = params.mean
            var = params.var
            critic_mean = params.critic_mean
            critic_var = params.critic_var
            count = params.count
        else:
            mean = jnp.mean(orig_obs, axis=0)
            var = jnp.var(orig_obs, axis=0)
            critic_mean = jnp.mean(critic_orig_obs, axis=0)
            critic_var = jnp.var(critic_orig_obs, axis=0)
            count = orig_obs.shape[0]
        state = DiffNormalizeVecObsEnvState(
            mean=mean,
            var=var,
            critic_mean=critic_mean,
            critic_var=critic_var,
            count=count,
            env_state=env_state,
            truncated=env_state.truncated,
            info=env_state.info,
        )
        norm_actor_obs = (orig_obs - state.mean) / jnp.sqrt(state.var + 1e-2)
        norm_critic_obs = (critic_orig_obs - state.critic_mean) / jnp.sqrt(
            state.critic_var + 1e-2
        )
        return (
            self._replace_orig_obs(obs, norm_actor_obs),
            self._replace_orig_obs(critic_obs, norm_critic_obs),
            state,
        )

    def step(self, key, state: DiffNormalizeVecObsEnvState, action):
        obs, critic_obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, action
        )
        orig_obs = obs["orig_obs"]
        critic_orig_obs = critic_obs["orig_obs"]

        new_mean, new_var = self._compute_stats(
            state.mean, state.var, state.count, orig_obs
        )
        new_critic_mean, new_critic_var = self._compute_stats(
            state.critic_mean, state.critic_var, state.count, critic_orig_obs
        )
        new_count = state.count + orig_obs.shape[0]

        state = DiffNormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            critic_mean=new_critic_mean,
            critic_var=new_critic_var,
            count=new_count,
            env_state=env_state,
            truncated=env_state.truncated,
            info=env_state.info,
        )

        norm_actor_obs = (orig_obs - state.mean) / jnp.sqrt(state.var + 1e-2)
        norm_critic_obs = (critic_orig_obs - state.critic_mean) / jnp.sqrt(
            state.critic_var + 1e-2
        )

        return (
            self._replace_orig_obs(obs, norm_actor_obs),
            self._replace_orig_obs(critic_obs, norm_critic_obs),
            state,
            reward,
            done,
            info,
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
