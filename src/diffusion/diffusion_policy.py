import jax
import numpy as np
import jax.numpy as jnp

from functools import partial

import optax
from gymnasium import spaces
from sac.policies import VectorCritic
from diffusion.common.utils import get_sampler_init
from diffusion.ud.ud_integrators import get_integrator as get_integrator_ud
from diffusion.od.od_integrators import get_integrator as get_integrator_od
from diffusion.od.od_sampling import sample as sample_od
from diffusion.ud.ud_sampling import sample as sample_ud
from common.policies import BaseJaxPolicy
from common.type_aliases import RLTrainState
from stable_baselines3.common.type_aliases import Schedule

from sac.utils import activation_fn


class DiffPol(BaseJaxPolicy):
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Box,
                 cfg,
                 squash_output: bool = True,
                 **kwargs,
                 ):
        super().__init__(observation_space,
                         action_space,
                         features_extractor=None,
                         features_extractor_kwargs=None,
                         squash_output=squash_output)
        self.cfg = cfg
        self.use_sde = False

    def build(self, key, lr_schedule: Schedule, qf_learning_rate: float):
        key, score_key, stat_distr_key, qf_key, dropout_key, stat_distr_bn_key, bn_key = jax.random.split(key, 7)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array([spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        a_dim = self.action_space.shape[0]
        buffer_obs_dim = obs.shape[1]  # This is the concatenated buffer observation dimension
        
        # Check if we have asymmetric observations
        actor_obs_dim = buffer_obs_dim  # Default to buffer size (symmetric case)
        critic_obs_dim = buffer_obs_dim  # Default to buffer size (symmetric case)
        
        # Check if the parent algorithm has asymmetric observation information
        if hasattr(self, '_asymmetric_obs_info') and self._asymmetric_obs_info:
            actor_obs_dim = self._asymmetric_obs_info.get('actor_obs_dim', buffer_obs_dim)
            critic_obs_dim = self._asymmetric_obs_info.get('critic_obs_dim', buffer_obs_dim)
            split_idx = self._asymmetric_obs_info.get('obs_split_idx', actor_obs_dim)
            print(f"Asymmetric observations detected in DiffPol:")
            print(f"  Buffer obs dim: {buffer_obs_dim}")
            print(f"  Actor obs dim: {actor_obs_dim}")
            print(f"  Critic obs dim: {critic_obs_dim}")
            print(f"  Split index: {split_idx}")
            
            # Create sample observations for network initialization
            actor_obs = obs[:, :split_idx]  # First part for actor
            critic_obs = obs[:, split_idx:] if split_idx < buffer_obs_dim else obs  # Remaining part for critic
        else:
            print(f"Using symmetric observations in DiffPol:")
            print(f"  Both actor and critic obs dim: {buffer_obs_dim}")
            actor_obs = obs
            critic_obs = obs
        
        print(f"DiffPol initialization - Actor obs shape: {actor_obs.shape}, Critic obs shape: {critic_obs.shape}")

        # initialize Q-function with critic observations
        self.qf = VectorCritic(
            dropout_rate=self.cfg.alg.critic.dropout_rate,
            use_layer_norm=self.cfg.alg.critic.use_layer_norm,
            use_batch_norm=self.cfg.alg.optimizer.bn,
            bn_warmup=self.cfg.alg.optimizer.bn_warmup,
            batch_norm_momentum=self.cfg.alg.optimizer.bn_momentum,
            batch_norm_mode=self.cfg.alg.optimizer.bn_mode,
            net_arch=self.cfg.alg.critic.hs,
            activation_fn=activation_fn[self.cfg.alg.critic.activation],
            n_critics=self.cfg.alg.critic.n_critics,
            n_atoms=self.cfg.alg.critic.n_atoms,
        )

        qf_init_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            critic_obs,  # Use critic observations for initialization
            action,
            train=False,
        )
        target_qf_init_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            critic_obs,  # Use critic observations for initialization
            action,
            train=False,
        )

        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=qf_init_variables["params"],
            batch_stats=qf_init_variables["batch_stats"],
            target_params=target_qf_init_variables["params"],
            target_batch_stats=target_qf_init_variables["batch_stats"],
            tx=optax.adam(
                learning_rate=qf_learning_rate,  # type: ignore[call-arg]
                **dict({
                    'b1': self.cfg.alg.optimizer.b1,
                    'b2': 0.999  # default
                }),
            ),
        )

        self.qf.apply = jax.jit(  # type: ignore[method-assign]
            self.qf.apply,
            static_argnames=("dropout_rate", "use_layer_norm",
                             "use_batch_norm", "batch_norm_momentum", "bn_mode"),
        )

        # Initialize actor
        key, diff_key = jax.random.split(key, 2)
        self.actor_model, self.actor_state = get_sampler_init(self.cfg.sampler.name)(diff_key, self.cfg, a_dim, actor_obs_dim)
        target_model_state = get_sampler_init(self.cfg.sampler.name)(diff_key, self.cfg, a_dim, actor_obs_dim)
        self.actor_target_model, self.target_actor_state = target_model_state
        if self.cfg.sampler.underdamped:
            self.integrator = get_integrator_ud(self.cfg, self.actor_model)
            self.target_integrator = get_integrator_ud(self.cfg, self.actor_target_model)
            sampler = sample_ud
        else:
            self.integrator = get_integrator_od(self.cfg, self.actor_model)
            self.target_integrator = get_integrator_od(self.cfg, self.actor_target_model)
            sampler = sample_od
        self.sampler = partial(sampler, integrator=self.integrator, diffusion_model=self.actor_model)
        self.target_sampler = partial(sampler, integrator=self.target_integrator,
                                      diffusion_model=self.actor_target_model)
        return key

    @staticmethod
    @partial(jax.jit, static_argnames=["sampler", "return_logprob", "stop_grad"])
    def sample_action(actor_state, actor_params, observations, key, sampler, stop_grad=False, return_logprob=False):
        out = sampler(key, actor_state, actor_params, observations, stop_grad=stop_grad)
        # terminal costs = prior log prob loss for od and prior log prob loss - momentum loss for ud
        final_action, running_costs, stochastic_costs, terminal_costs, a_t, v_t = out
        # final_action: (1, 6) for n_envs=1
        return final_action, running_costs, stochastic_costs, terminal_costs, a_t, v_t

    @staticmethod
    def multi_sample(actor_state, actor_params, observations, key, sampler, stop_grad=False):
        x, y, z = observations.shape
        obs_reshaped = jnp.transpose(observations, (0, 2, 1)).reshape((x * z, y))
        keys = jax.random.split(key, x * z)

        def sample_single(k, obs):
            return DiffPol.sample_action(actor_state, actor_params, obs[None], k, sampler, stop_grad=stop_grad)

        batched_sample = jax.vmap(sample_single, in_axes=(0, 0))
        final_action, running_costs, stochastic_costs, terminal_costs, a_t, v_t = batched_sample(keys, obs_reshaped)

        a_dim = final_action.shape[-1]
        T = a_t.shape[1]  # time dimension

        # Reshape back to (x, z, ...) then transpose as needed
        final_action = final_action.reshape(x, z, a_dim).transpose(0, 2, 1)  # (x, a_dim, z)
        running_costs = running_costs.reshape(x, z)
        stochastic_costs = stochastic_costs.reshape(x, z)
        terminal_costs = terminal_costs.reshape(x, z)

        a_t = 0
        v_t = 0

        return final_action, running_costs, stochastic_costs, terminal_costs, a_t, v_t

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        
        # Split concatenated observation if using asymmetric observations
        if hasattr(self, '_asymmetric_obs_info') and self._asymmetric_obs_info:
            split_idx = self._asymmetric_obs_info.get('obs_split_idx', observation.shape[-1])
            actor_obs = observation[..., :split_idx]
        else:
            actor_obs = observation
            
        actions, *_ = DiffPol.sample_action(self.actor_state, self.actor_state.params, actor_obs, self.noise_key,
                                            self.sampler)
        # return actions[0] # in the setting n_envs=1, actions (1, 6) so we return actions[0] to get (6,)
        return actions # in the setting n_envs=1, actions (1, 6) so we return actions[0] to get (6,)

    def _predict2(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        actions, _, _, _, la, _ = DiffPol.sample_action(self.actor_state, self.actor_state.params, observation,
                                                        self.noise_key, self.sampler)
        actions = (actions, la)
        return actions

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    def predict_critic(self, observation: np.ndarray, action: np.ndarray) -> np.ndarray:

        if not self.use_sde:
            self.reset_noise()
        def Q(params, batch_stats, o, a, dropout_key):
            return self.qf_state.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                o, a,
                rngs={"dropout": dropout_key},
                train=False
            )

        return jax.jit(Q)(
            self.qf_state.params,
            self.qf_state.batch_stats,
            observation,
            action,
            self.noise_key,
        )
        
    def get_critic_obs_from_info(self, infos):
        """Extract critic observations from environment info dict"""
        if not hasattr(self, '_asymmetric_obs_info'):
            return None
            
        critic_obs_list = []
        for info in infos:
            if 'critic_obs' in info:
                critic_obs_list.append(info['critic_obs'])
        
        if critic_obs_list:
            return np.array(critic_obs_list)
        return None