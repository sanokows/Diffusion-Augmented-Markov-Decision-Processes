from flax import nnx
import jax.numpy as jnp

class ControlNetwork(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        num_layers: int = 2,
        num_hid: int = 64,
        num_time_hid: int = 32,
        num_time_out: int = 16,
        outer_clip: float = 1e4,
        inner_clip: float = 1e2,
        weight_init: float = 1e-8,
        bias_init: float = 0.0,
        layer_norm: bool = False,
        layer_norm_type: str = "LayerNorm",
        *,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.layer_norm = layer_norm
        self.layer_norm_type = layer_norm_type
        self.num_layers = num_layers
        self.num_hid = num_hid
        self.num_time_hid = num_time_hid
        self.num_time_out = num_time_out
        self.outer_clip = outer_clip
        self.inner_clip = inner_clip
        self.weight_init = weight_init
        self.bias_init = bias_init

        # Initialize timestep parameters
        self.timestep_phase = nnx.Param(jnp.zeros((1, self.num_time_hid)))
        # Store timestep_coeff as a Variable (non-trainable parameter)
        self.timestep_coeff = nnx.Variable(
            jnp.linspace(start=0.1, stop=100, num=self.num_time_hid)[None]
        )

        # Time encoder network
        self.time_coder_state = nnx.Sequential(
            nnx.Linear(self.num_time_hid * 2, self.num_time_hid, rngs=rngs),
            nnx.gelu,
            nnx.Linear(self.num_time_hid, self.num_time_out, rngs=rngs),
        )

        # State-time network
        if self.layer_norm:
            layers = []
            layers.extend(
                [
                    nnx.Linear(
                        self.action_dim + self.observation_dim + self.num_time_out,
                        self.num_hid,
                        rngs=rngs,
                    ),
                    nnx.gelu,
                ]
            )
            for _ in range(self.num_layers - 2):
                layers.extend(
                    [
                        nnx.Linear(self.num_hid, self.num_hid, rngs=rngs),
                        getattr(nnx, self.layer_norm_type)(self.num_hid, rngs=rngs),
                        nnx.gelu,
                    ]
                )
            # Output layer with custom initialization
            output_layer = nnx.Linear(self.num_hid, self.action_dim, rngs=rngs)
            # Apply custom initialization
            output_layer.kernel.value = output_layer.kernel.value * self.weight_init
            output_layer.bias.value = (
                jnp.zeros_like(output_layer.bias.value) + self.bias_init
            )
            layers.append(output_layer)
            self.state_time_net = nnx.Sequential(*layers)
        else:
            layers = []
            layers.extend(
                [
                    nnx.Linear(
                        self.action_dim + self.observation_dim + self.num_time_out,
                        self.num_hid,
                        rngs=rngs,
                    ),
                    nnx.gelu,
                ]
            )
            for _ in range(self.num_layers - 2):
                layers.extend(
                    [
                        nnx.Linear(self.num_hid, self.num_hid, rngs=rngs),
                        nnx.gelu,
                    ]
                )
            # Output layer with custom initialization
            output_layer = nnx.Linear(self.num_hid, self.action_dim, rngs=rngs)
            # Apply custom initialization
            output_layer.kernel.value = output_layer.kernel.value * self.weight_init
            output_layer.bias.value = (
                jnp.zeros_like(output_layer.bias.value) + self.bias_init
            )
            layers.append(output_layer)
            self.state_time_net = nnx.Sequential(*layers)

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin(
            (self.timestep_coeff.value * timesteps) + self.timestep_phase.value
        )
        cos_embed_cond = jnp.cos(
            (self.timestep_coeff.value * timesteps) + self.timestep_phase.value
        )
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(self, actions, observations, time):
        time_emb = self.get_fourier_features(time)
        if len(actions.shape) == 1:
            time_emb = time_emb[0]
        t_net = self.time_coder_state(time_emb)

        extended_input = jnp.concatenate((actions, observations, t_net), axis=-1)
        out_state = self.state_time_net(extended_input)
        out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)
        return out_state