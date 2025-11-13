import math
import torch
from torch import nn
from torch.distributions.normal import Normal

from src.torchrl.reppo_util import hl_gauss
from src.networks.torch_utils import inverse_softplus, sample_kernel, log_prob_kernel, check_stop_grad 

# from .helpers import SinusoidalPosEmb, cosine_beta_schedule, linear_beta_schedule, vp_beta_schedule, extract, Losses
# from .diffusion_utils import Progress, Silent

def get_activation(name):
    if name == "gelu":
        return nn.GELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "swish":
        return nn.SiLU()
    elif name is None:
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation: {name}")


def normed_activation_layer(
    in_features, out_features, use_norm=True, activation="swish", device=None
):
    layers = [nn.Linear(in_features, out_features, device=device)]
    if use_norm:
        layers.append(nn.RMSNorm([out_features], device=device))
        # layers.append(nn.LayerNorm([out_features], device=device)) # for torch 2.4.0
    if activation is not None:
        layers.append(get_activation(activation))
    return nn.Sequential(*layers)


class FCNN(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_dim=256,
        hidden_activation="swish",
        output_activation=None,
        use_norm=True,
        use_output_norm=False,
        layers=2,
        input_activation=False,
        device=None,
    ):
        super().__init__()
        net = []
        if layers == 1:
            net.append(
                normed_activation_layer(
                    in_features,
                    out_features,
                    use_norm=use_output_norm,
                    activation=output_activation,
                    device=device,
                )
            )
        else:
            if input_activation:
                net.append(get_activation(hidden_activation))
            net.append(
                normed_activation_layer(
                    in_features,
                    hidden_dim,
                    use_norm=use_norm,
                    activation=hidden_activation,
                    device=device,
                )
            )
            for _ in range(layers - 2):
                net.append(
                    normed_activation_layer(
                        hidden_dim,
                        hidden_dim,
                        use_norm=use_norm,
                        activation=hidden_activation,
                        device=device,
                    )
                )
            net.append(
                normed_activation_layer(
                    hidden_dim,
                    out_features,
                    use_norm=use_output_norm,
                    activation=output_activation,
                    device=device,
                )
            )
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class CriticNetwork(nn.Module):
    def __init__(
        self,
        n_obs,
        n_act,
        hidden_dim=256,
        use_norm=True,
        use_encoder_norm=False,
        encoder_layers=1,
        head_layers=1,
        pred_layers=1,
        device=None,
    ):
        super().__init__()
        self.feature_module = FCNN(
            in_features=n_obs + n_act,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation="swish",
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=use_encoder_norm,
            layers=encoder_layers,
            device=device,
        )
        self.critic_module = FCNN(
            in_features=hidden_dim,
            out_features=1,
            hidden_dim=hidden_dim,
            hidden_activation="swish",
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=head_layers,
            device=device,
        )
        self.pred_module = FCNN(
            in_features=hidden_dim,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation="swish",
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=pred_layers,
            device=device,
        )

    def features(self, obs, action):
        state = torch.cat([obs, action], dim=-1)
        return self.feature_module(state)

    def critic_head(self, features):
        return self.critic_module(features)

    def critic(self, obs, action):
        features = self.features(obs, action)
        return self.critic_head(features)

    def forward(self, obs, action):
        features = self.features(obs, action)
        return self.pred_module(features)


class Critic(nn.Module):
    def __init__(
        self,
        n_obs,
        n_act,
        num_atoms: int,
        vmin: float,
        vmax: float,
        hidden_dim=256,
        use_norm=True,
        use_encoder_norm=False,
        encoder_layers=1,
        head_layers=1,
        pred_layers=1,
        device=None,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.vmin = vmin
        self.vmax = vmax
        self.hidden_dim = hidden_dim
        self.feature_module = FCNN(
            in_features=n_obs + n_act,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation="swish",
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=use_encoder_norm,
            layers=encoder_layers,
            device=device,
        )
        self.critic_module = FCNN(
            in_features=hidden_dim,
            out_features=num_atoms,
            hidden_dim=hidden_dim,
            hidden_activation="swish",
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            input_activation=True,
            layers=head_layers,
            device=device,
        )
        self.pred_module = FCNN(
            in_features=hidden_dim,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation="swish",
            output_activation=None,
            use_norm=use_norm,
            input_activation=True,
            use_output_norm=False,
            layers=pred_layers,
            device=device,
        )
        self.values = torch.linspace(
            vmin, vmax, num_atoms, device=device, dtype=torch.float32
        )
        zeros = hl_gauss(
            torch.zeros(1, device=device), self.vmin, self.vmax, self.num_atoms
        )
        zeros.requires_grad = True
        self.zero_dist = nn.Parameter(
            hl_gauss(
                torch.zeros(1, device=device), self.vmin, self.vmax, self.num_atoms
            )
        )

    def forward(self, obs, action):
        inp = torch.cat([obs, action], dim=-1)
        features = self.feature_module(inp)
        next_pred = self.pred_module(features)
        logits = self.critic_module(features) + 40.9 * self.zero_dist
        value_cats = torch.softmax(logits, dim=-1)
        value = value_cats @ self.values
        return value, logits, next_pred, features


class Actor(nn.Module):
    def __init__(
        self,
        n_obs,
        n_act,
        ent_start: float,
        kl_start: float,
        hidden_dim=256,
        use_norm=True,
        layers=2,
        min_std=0.1,
        device=None,
    ):
        super().__init__()
        self.model = FCNN(
            in_features=n_obs,
            out_features=2 * n_act,
            hidden_dim=hidden_dim,
            hidden_activation="swish",
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=layers,
            device=device,
        )
        self.log_temp = nn.Parameter(
            torch.log(torch.tensor(ent_start, device=device, dtype=torch.float32))
        )
        self.log_lagrange = nn.Parameter(
            torch.log(torch.tensor(kl_start, device=device, dtype=torch.float32))
        )
        self.min_std = min_std

    def forward(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        x = self.model(obs)
        mean, log_std = torch.split(x, x.shape[-1] // 2, dim=-1)
        std = torch.exp(log_std) + self.min_std
        pi = Normal(mean, std, validate_args=False)

        transformed_pi = torch.distributions.TransformedDistribution(
            pi, [torch.distributions.TanhTransform()]
        )
        return (
            transformed_pi,
            torch.tanh(mean),
            torch.exp(self.log_temp),
            torch.exp(self.log_lagrange),
        )

class ControlNetwork(nn.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        num_layers: int = 2,
        num_hid: int = 64,
        num_time_hid: int = 64,
        num_time_out: int = 64,
        outer_clip: float = 1e4,
        inner_clip: float = 1e2,
        weight_init: float = 1e-8,
        bias_init: float = 0.0,
        layer_norm: bool = False,
        layer_norm_type: str = "LayerNorm",
        state_encoder: nn.Module = None,
        device=None,
    ):
        super().__init__()
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
        self.state_encoder = state_encoder

        # Initialize timestep parameters
        self.timestep_phase = nn.Parameter(torch.zeros(1, self.num_time_hid, device=device))
        # Store timestep_coeff as a buffer (non-trainable parameter)
        self.register_buffer(
            'timestep_coeff', 
            torch.linspace(0.1, 100, self.num_time_hid, device=device).unsqueeze(0)
        )

        # Time encoder network
        self.time_coder_state = nn.Sequential(
            nn.Linear(self.num_time_hid * 2, self.num_time_hid, device=device),
            nn.GELU(),
            nn.Linear(self.num_time_hid, self.num_time_out, device=device),
        )

        # State-time network
        layers = []
        layers.extend(
            [
                nn.Linear(
                    (
                        self.action_dim + self.num_hid + self.num_time_out
                        if state_encoder
                        else self.action_dim + self.observation_dim + self.num_time_out
                    ),
                    self.num_hid,
                    device=device,
                ),
                nn.GELU(),
            ]
        )

        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.num_hid, self.num_hid, device=device))
            if self.layer_norm:
                if self.layer_norm_type == "LayerNorm":
                    layers.append(nn.LayerNorm(self.num_hid, device=device))
                elif self.layer_norm_type == "RMSNorm":
                    layers.append(nn.RMSNorm(self.num_hid, device=device))
            layers.append(nn.GELU())

        # Output layer with custom initialization
        output_layer = nn.Linear(self.num_hid, self.action_dim, device=device)
        # Apply custom initialization
        with torch.no_grad():
            output_layer.weight.data *= self.weight_init
            output_layer.bias.data.fill_(self.bias_init)
        layers.append(output_layer)

        self.state_time_net = nn.Sequential(*layers)

    def get_fourier_features(self, timesteps):
        sin_embed_cond = torch.sin(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        cos_embed_cond = torch.cos(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        return torch.cat([sin_embed_cond, cos_embed_cond], dim=-1)

    def forward(self, actions, observations, time):
        time_emb = self.get_fourier_features(time)
        t_net = self.time_coder_state(time_emb)

        # repeat to match actions
        t_net = t_net.repeat_interleave(actions.size(0), dim=0)
        if self.state_encoder:
            encoded_observations = self.state_encoder(observations)
        else:
            encoded_observations = observations

        extended_input = torch.cat((actions, encoded_observations, t_net), dim=-1)
        out_state = self.state_time_net(extended_input)
        out_state = torch.clamp(out_state, -self.outer_clip, self.outer_clip)
        return out_state

class DiffusionModel(nn.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        fwd_model: nn.Module = None,
        bwd_model: nn.Module = None,
        diff_steps: int = 8,
        init_std: float = 2.5,
        friction: float = 1.0,
        per_dim_friction: bool = True,
        dt: float = 0.01,
        learn_dt: bool = True,
        per_step_dt: bool = False,
        learn_prior: bool = False,
        learn_betas: bool = False,
        learn_friction: bool = True,
        learn_mass_matrix: bool = False,
        dt_schedule: callable = None,
        device=None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.diff_steps = diff_steps
        self.init_std = init_std
        self.fwd_model = fwd_model
        self.bwd_model = bwd_model
        self.learn_prior = learn_prior
        self.learn_friction = learn_friction
        self.learn_mass_matrix = learn_mass_matrix
        self.learn_dt = learn_dt
        self.learn_betas = learn_betas
        self.per_step_dt = per_step_dt
        self.dt_schedule = dt_schedule
        
        # Learnable parameters (converted from the params dict)
        self.betas = nn.Parameter(torch.ones(diff_steps, device=device))
        self.prior_mean = nn.Parameter(torch.zeros(action_dim, device=device))
        self.prior_std = nn.Parameter(torch.ones(action_dim, device=device) * inverse_softplus(torch.tensor(init_std)))
        self.mass_std = nn.Parameter(torch.ones(1, device=device) * inverse_softplus(torch.tensor(1.0)))

        # Initialize dt parameters
        if per_step_dt:
            steps = torch.arange(diff_steps, dtype=torch.float32, device=device)
            if dt_schedule is not None:
                dt_values = dt * dt_schedule(steps)
            else:
                dt_values = dt * torch.ones_like(steps)
            self.dt = nn.Parameter(inverse_softplus(dt_values))
        else:
            self.dt = nn.Parameter(torch.ones(1, device=device) * inverse_softplus(torch.tensor(dt)))
        
        # Initialize friction parameters
        if per_dim_friction:
            self.friction = nn.Parameter(torch.ones(action_dim, device=device) * inverse_softplus(torch.tensor(friction)))
        else:
            self.friction = nn.Parameter(torch.ones(1, device=device) * inverse_softplus(torch.tensor(friction)))

    def prior_sampler(self, n_samples, stop_grad, device=None):
        """Sample from the prior distribution."""
        device = device or self.prior_mean.device
        mean = self.prior_mean if self.learn_prior else torch.zeros(self.action_dim, device=device)
        std = torch.nn.functional.softplus(self.prior_std) if self.learn_prior else torch.ones(self.action_dim, device=device) * self.init_std
        dist = torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=std), 1
        )
        samples = dist.rsample((n_samples,))
        
        return samples

    def prior_log_prob(self, x):
        """Compute log probability under the prior."""
        if self.learn_prior:
            mean = self.prior_mean
            std = torch.nn.functional.softplus(self.prior_std)
        else:
            mean = torch.zeros(self.action_dim, device=x.device)
            std = torch.ones(self.action_dim, device=x.device) * self.init_std
        
        dist = torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)  # diagonal Gaussian
        return dist.log_prob(x)

    def delta_t_fn(self, step):
        """Time step function."""
        if self.per_step_dt:
            dt = self.dt[step.long()] if self.learn_dt else self.dt[step.long()].detach()
            return torch.nn.functional.softplus(dt)
        else:
            dt = self.dt if self.learn_dt else self.dt.detach()
            dt_val = torch.nn.functional.softplus(dt)
            if self.dt_schedule is not None:
                return dt_val * self.dt_schedule(step)
            else:
                return dt_val

    def friction_fn(self, step):
        """Friction coefficient function."""
        friction = torch.nn.functional.softplus(self.friction)
        return friction if self.learn_friction else friction.detach()

    def mass_fn(self):
        """Mass function."""
        mass_std = torch.nn.functional.softplus(self.mass_std)
        return mass_std if self.learn_mass_matrix else mass_std.detach()

    def drift_fn(self, step, x):
        """Drift function for diffusion (gradient of prior log prob)."""
        # Fall back to analytical gradient: ∇_x log p(x) = -(x-μ)/σ²
        mean = self.prior_mean if self.learn_prior else torch.zeros(self.action_dim, device=x.device)
        std = torch.nn.functional.softplus(self.prior_std) if self.learn_prior else torch.ones(self.action_dim, device=x.device) * self.init_std
        grad = -(x - mean) / (std ** 2)
        return grad

    def forward_model(self, step, x, obs):
        """Forward model function."""
        if self.fwd_model is not None:
            return self.fwd_model(x, obs, step)
        else:
            return torch.zeros_like(x)

    def backward_model(self, step, x, obs, aux=None):
        """Backward model function."""
        if self.bwd_model is not None:
            return self.bwd_model(x, obs, step)
        else:
            return torch.zeros_like(x)

    def diffusion_coef(self, step):
        """Diffusion coefficient function."""
        # Simple implementation - this might need to be adjusted based on your specific model
        return torch.ones_like(step) if isinstance(step, torch.Tensor) else torch.tensor(1.0)

class DIMEActor(nn.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        diffusion_model: nn.Module,
        kl_start: float = 0.1,
        ent_start: float = 0.1,
        asymmetric_obs: bool = False,
        device=None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.diffusion_model = diffusion_model
        self.asymmetric_obs = asymmetric_obs

        self.log_temperature = nn.Parameter(
            torch.ones(1, device=device) * math.log(ent_start)
        )
        self.log_lagrangian = nn.Parameter(
            torch.ones(1, device=device) * math.log(kl_start)
        )

    def sample(
        self,
        obs: torch.Tensor,
        stop_grad: bool = False,
        ode: bool = False,
        ode_coef: float = 1.0,
    ) -> torch.Tensor:
        """Sample actions from the diffusion model."""
        if self.asymmetric_obs:
            assert (
                isinstance(obs, dict) and "state" in obs
            ), "State must be provided for actor."
            obs = obs["state"]
        bs, *_ = obs.shape
        init_x = self.diffusion_model.prior_sampler(bs, stop_grad=stop_grad)
        if stop_grad:
            init_x = init_x.detach()

        log_w = torch.zeros(bs, device=obs.device, dtype=torch.float32)
        x = init_x
        for step in torch.arange(0, self.diffusion_model.diff_steps, dtype=torch.float32):
            # integrator jax
            dt = self.diffusion_model.delta_t_fn(step)
            sigma_square = 1. / self.diffusion_model.friction_fn(step)
            eta = dt * sigma_square
            scale = torch.sqrt(2 * eta)

            # Forward kernel
            drift = self.diffusion_model.drift_fn(step, x)
            fwd_mean = x + eta * (drift + (ode_coef * self.diffusion_model.forward_model(step, x, obs))) if ode else x + eta * (drift + self.diffusion_model.forward_model(step, x, obs))
            x_new = fwd_mean if ode else sample_kernel(check_stop_grad(fwd_mean, stop_grad) if stop_grad else fwd_mean, scale)

            # Backward kernel
            drift_new = self.diffusion_model.drift_fn(step + 1, x_new)
            bwd_mean = x_new + eta * (drift_new + self.diffusion_model.backward_model(step + 1, x_new, obs))

            # Evaluate kernels
            fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
            bwd_log_prob = log_prob_kernel(x, bwd_mean, scale)

            # Update weight and return
            log_w += bwd_log_prob - fwd_log_prob
            x = x_new

        final_x = x
        terminal_costs = self.diffusion_model.prior_log_prob(init_x)

        # Compute log determinant of Jacobian for tanh transformation
        tanh_transform = torch.distributions.TanhTransform()
        tanh_log_det_jac = tanh_transform.log_abs_det_jacobian(final_x, tanh_transform(final_x)).sum(dim=-1)

        running_cost = -(log_w + tanh_log_det_jac)

        # Apply tanh transformation to get final action
        final_action = tanh_transform(final_x)
        stochastic_costs = torch.zeros_like(running_cost)

        # return final_action, final_x, terminal_costs, running_cost, stochastic_costs, log_w
        return final_action, running_cost, stochastic_costs, terminal_costs

    def kl_div(
        self, 
        obs: torch.Tensor, 
        target_actor: nn.Module, 
        n_samples: int,
        stop_grad: bool = False
    ) -> torch.Tensor:
        """Compute KL divergence between current and old diffusion models."""
        if self.asymmetric_obs:
            assert (
                isinstance(obs, dict) and "state" in obs
            ), "State must be provided for actor."
            obs = obs["state"]

        # repeat obs for n_samples from [bs, ...] to [bs * n_samples, ...]
        if n_samples > 1:
            obs = obs.repeat_interleave(n_samples, dim=0)
        bs, action_dim = obs.shape

        init_x = self.diffusion_model.prior_sampler(bs, stop_grad=stop_grad)
        if stop_grad:
            init_x = init_x.detach()

        log_w = torch.zeros(bs, device=obs.device, dtype=torch.float32)
        x = init_x
        for step in torch.arange(0, self.diffusion_model.diff_steps, dtype=torch.float32):
            # integrator jax
            dt = self.diffusion_model.delta_t_fn(step)
            sigma_square = 1. / self.diffusion_model.friction_fn(step)
            eta = dt * sigma_square
            scale = torch.sqrt(2 * eta)

            # Forward kernel
            drift = self.diffusion_model.drift_fn(step, x)
            fwd_mean = x + eta * (drift + self.diffusion_model.forward_model(step, x, obs))
            old_fwd_mean = x + eta * (drift + target_actor.diffusion_model.forward_model(step, x, obs))

            # x = sample_kernel(check_stop_grad(fwd_mean, stop_grad) if stop_grad else fwd_mean, scale)
            x_new = sample_kernel(check_stop_grad(old_fwd_mean, stop_grad) if stop_grad else old_fwd_mean, scale)
            # x_new = sample_kernel(key, check_stop_grad(fwd_mean, stop_grad) if stop_grad else fwd_mean, scale)

            # Evaluate kernels
            fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
            old_fwd_log_prob = log_prob_kernel(x_new, old_fwd_mean, scale)

            # Update weight and return
            log_w += old_fwd_log_prob - fwd_log_prob
            x = x_new

        final_x = x

        # Apply tanh transformation to get final action
        final_action = torch.tanh(final_x)
        log_ratios = log_w

        # take average over n_samples
        if n_samples > 1:
            log_ratios = log_ratios.view(-1, n_samples).mean(dim=-1)

        return final_action, log_ratios

    def forward(
        self,
        obs: torch.Tensor,
        stop_grad: bool = False,
        ode: bool = False,
        ode_coef: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass - sample actions from diffusion model."""
        final_action, running_cost, stochastic_costs, terminal_costs = self.sample(
            obs, stop_grad=stop_grad, ode=ode, ode_coef=ode_coef
        )
        return (
            final_action,
            running_cost,
            stochastic_costs,
            terminal_costs,
        )

    def temperature(self) -> torch.Tensor:
        """Get current temperature value."""
        return torch.exp(self.log_temperature)
    
    def lagrangian(self) -> torch.Tensor:
        """Get current lagrangian multiplier value."""
        return torch.exp(self.log_lagrangian)


class StochasticPolicy(nn.Module):
    def __init__(self, actor: Actor, normalizer: nn.Module = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = actor
        self.normalizer = normalizer

    def forward(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        if self.normalizer:
            obs = self.normalizer(obs)
        return self.actor(obs)


class TD3DeterministicPolicy(nn.Module):
    def __init__(
        self,
        n_obs,
        n_act,
        hidden_dim=256,
        use_norm=True,
        layers=2,
        device=None,
    ):
        super().__init__()
        self.model = FCNN(
            in_features=n_obs,
            out_features=2 * n_act,
            hidden_dim=hidden_dim,
            hidden_activation="swish",
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=layers,
            device=device,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.model(obs)
        mean, _ = torch.split(x, x.shape[-1] // 2, dim=-1)
        return torch.tanh(mean)
