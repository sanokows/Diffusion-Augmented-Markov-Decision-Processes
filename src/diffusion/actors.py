from flax import linen as nn
import jax.numpy as jnp
from typing import Sequence
from jax._src.nn.functions import softplus
import tensorflow_probability
from sympy.physics.units.systems.si import dimex

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

def mish(x):
    return x * jnp.tanh(softplus(x))

class PISGRADNet(nn.Module):
    dim: int
    obs_dim: int

    net_arch: list
    outer_clip: float = 1e4
    inner_clip: float = 1e2
    beta_max = 100
    beta_min = 0.01
    weight_init: float = 1e-8
    bias_init: float = 0.

    def setup(self):
        num_hid = self.net_arch[0]
        num_layers = len(self.net_arch)

        self.timestep_phase = self.param('timestep_phase', nn.initializers.zeros_init(), (1, num_hid))
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=num_hid)[None]

        self.time_coder_state = nn.Sequential([
            nn.Dense(num_hid),
            nn.gelu,
            # nn.relu,
            # nn.Dense(num_hid),
            nn.Dense(128),
        ])

        # self.time_coder_state = nn.Sequential([
        #     nn.Dense(num_hid),
        #     # nn.gelu,
        #     nn.relu,
        #     # mish,
        #     nn.Dense(self.dim+self.obs_dim),
        # ])

        # self.timestep_phase = self.param('timestep_phase', nn.initializers.zeros_init(), (1,
        #                                                                                   self.obs_dim+self.dim))
        # self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.obs_dim+self.dim)[None]
        #
        # self.time_coder_state = nn.Sequential([
        #     nn.Dense(num_hid),
        #     nn.gelu,
        #     nn.Dense(self.obs_dim+self.dim),
        # ])
        #
        # self.time_coder_grad = nn.Sequential([nn.Dense(self.num_hid)] + [nn.Sequential(
        #     [nn.gelu, nn.Dense(num_hid)]) for _ in range(num_layers)] + [
        #                                          nn.Dense(self.dim,
        #                                                   kernel_init=nn.initializers.constant(self.weight_init),
        #                                                   bias_init=nn.initializers.constant(self.bias_init))])
        #
        ######
        #
        self.state_time_net = nn.Sequential([nn.Sequential(
            # [nn.Dense(num_hid), nn.gelu]) for _ in range(num_layers)] + [
            [nn.Dense(num_hid), nn.relu]) for _ in range(num_layers)] + [
            # [nn.Dense(num_hid), mish]) for _ in range(num_layers)] + [
                                                nn.Dense(self.dim,
                                                         kernel_init=nn.initializers.constant(1e-8),
                                                         bias_init=nn.initializers.zeros_init()
                                                         )])

        self.obs_encoder = nn.Sequential([
            # nn.Dense(num_hid, kernel_init=nn.initializers.constant(1e-8),
            nn.Dense(128, kernel_init=nn.initializers.constant(1e-8),
                     bias_init=nn.initializers.constant(1.0)),
            ])

        self.act_encoder = nn.Sequential([
            # nn.Dense(num_hid, kernel_init=nn.initializers.constant(1e-8),
            nn.Dense(128, kernel_init=nn.initializers.constant(1e-8),
                     bias_init=nn.initializers.constant(1.0)),
            ])


    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        cos_embed_cond = jnp.cos(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(self, input_array, obs_array, time_array, train=False):
        time_array_emb = self.get_fourier_features(time_array)
        if len(input_array.shape) == 1:
            time_array_emb = time_array_emb[0]
        t_net1 = self.time_coder_state(time_array_emb)
        # extended_input = jnp.concatenate((input_array, obs_array, t_net1), axis=-1)
        encoded_obs = self.obs_encoder(obs_array)
        encoded_acts = self.act_encoder(input_array)
        extended_input = jnp.concatenate((encoded_acts, encoded_obs, t_net1), axis=-1)
        out_state = self.state_time_net(extended_input)
        out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
        return out_state


class StatDistrActor(nn.Module):
    net_arch: Sequence[int]
    action_dim: int
    log_std_min: float = -2
    log_std_max: float = 2

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    @nn.compact
    # type: ignore[name-defined]
    def __call__(self, x: jnp.ndarray, train) -> tfd.Distribution:

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = nn.relu(x)

        mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.constant(1e-8),
                     bias_init=nn.initializers.constant(0.0))(x)
        log_std = nn.Dense(self.action_dim, kernel_init=nn.initializers.constant(1e-8),
                     bias_init=nn.initializers.constant(0.0))(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        # dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.ones(self.action_dim)*1.0)
        return dist