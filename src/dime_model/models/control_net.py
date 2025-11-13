import jax.numpy as jnp
from flax import linen as nn


class ControlNetwork(nn.Module):
    dim: int
    layer_norm: bool = False
    layer_norm_type: str = "LayerNorm"
    time_coder_out: int = 64

    num_layers: int = 2
    num_hid: int = 64
    num_time_hid: int = 16
    outer_clip: float = 1e4
    inner_clip: float = 1e2

    weight_init: float = 1e-8
    bias_init: float = 0.

    def setup(self):
        self.timestep_phase = self.param('timestep_phase', nn.initializers.zeros_init(), (1, self.num_time_hid))
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_time_hid)[None]

        self.time_coder_state = nn.Sequential([
            nn.Dense(self.num_time_hid),
            nn.gelu,
            nn.Dense(self.time_coder_out),
        ])

        if self.layer_norm:
            self.state_time_net = nn.Sequential([nn.Sequential([nn.Dense(self.num_hid), 
                                                                getattr(nn, self.layer_norm_type)(), 
                                                                nn.gelu]) 
                                                 for _ in range(self.num_layers)] 
                                                 + [nn.Dense(self.dim, kernel_init=nn.initializers.constant(1e-8),
                                                             bias_init=nn.initializers.zeros_init())])
        else:
            self.state_time_net = nn.Sequential([nn.Sequential([nn.Dense(self.num_hid), nn.gelu]) 
                                                 for _ in range(self.num_layers)] 
                                                + [nn.Dense(self.dim, kernel_init=nn.initializers.constant(1e-8),
                                                            bias_init=nn.initializers.zeros_init())])

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        cos_embed_cond = jnp.cos(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(self, input_array, obs_array, time_array):
        time_array_emb = self.get_fourier_features(time_array)
        if len(input_array.shape) == 1:
            time_array_emb = time_array_emb[0]

        t_net1 = self.time_coder_state(time_array_emb)

        extended_input = jnp.concatenate((input_array, obs_array, t_net1), axis=-1)
        out_state = self.state_time_net(extended_input)
        out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
        return out_state
