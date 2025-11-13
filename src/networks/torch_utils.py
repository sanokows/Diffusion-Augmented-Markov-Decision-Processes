import torch
import torch.nn.functional as F
import numpy as np
import wandb
import matplotlib.pyplot as plt


def inverse_softplus(x):
    """Numerically stable implementation of inverse softplus"""
    # Threshold above which the approximation log(e^x - 1) ≈ x is used
    threshold = 20.0
    return torch.where(x > threshold, x, torch.log(torch.expm1(x)))


def check_stop_grad(expression, stop_grad):
    """Stop gradients conditionally"""
    return expression.detach() if stop_grad else expression


def sample_kernel(mean, scale, device=None):
    """Sample from a normal distribution"""
    device = device or mean.device
    # eps = torch.randn(mean.shape[0], device=device)
    eps = torch.randn_like(mean, device=device)
    return mean + scale * eps


def log_prob_kernel(x, mean, scale):
    """Compute log probability under normal distribution"""
    dist = torch.distributions.Independent(
        torch.distributions.Normal(loc=mean, scale=scale), 1
    )
    return dist.log_prob(x)


def avg_list_entries(list_data, num):
    """Average consecutive entries in a list"""
    assert len(list_data) >= num
    print(range(0, len(list_data) - num))
    return [sum(list_data[i:i + num]) / float(num) for i in range(0, len(list_data) - num + 1)]


def reverse_transition_params(transition_params):
    """Reverse parameters along the first axis"""
    def reverse_tensor(tensor):
        if isinstance(tensor, torch.Tensor):
            return torch.flip(tensor, dims=[0])
        return tensor
    
    if isinstance(transition_params, dict):
        return {k: reverse_transition_params(v) for k, v in transition_params.items()}
    elif isinstance(transition_params, (list, tuple)):
        return type(transition_params)(reverse_transition_params(item) for item in transition_params)
    else:
        return reverse_tensor(transition_params)


def interpolate_values(values, X):
    """Compute interpolated values"""
    # Compute the interpolated values
    interpolated_values = [X] + [X + (X / 2 - X) * t for t in values[1:-1]] + [X / 2]
    return interpolated_values


def flattened_traversal(fn):
    """Create a mask function for nested dictionaries"""
    def mask(data):
        def flatten_dict(d, parent_key='', sep='/'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        def unflatten_dict(flat_dict, sep='/'):
            result = {}
            for key, value in flat_dict.items():
                parts = key.split(sep)
                d = result
                for part in parts[:-1]:
                    if part not in d:
                        d[part] = {}
                    d = d[part]
                d[parts[-1]] = value
            return result
        
        flat = flatten_dict(data)
        masked_flat = {k: fn(k, v) for k, v in flat.items()}
        return unflatten_dict(masked_flat)
    
    return mask


def plot_annealing(model_state, cfg):
    """Plot annealing schedule"""
    if cfg.use_wandb:
        fig, ax = plt.subplots()
        b = F.softplus(model_state.params['params']['betas'])
        b = torch.cumsum(b, dim=0) / torch.sum(b)
        
        ax.plot(b.detach().cpu().numpy())
        return {"figures/annealing": [wandb.Image(fig)]}
    else:
        return {}


def plot_timesteps(diffusion_model, model_state, cfg):
    """Plot timesteps"""
    if cfg.use_wandb:
        steps = torch.arange(cfg.algorithm.num_steps, device=next(iter(model_state.params.values())).device)
        dts = torch.stack([diffusion_model.delta_t_fn(step) for step in steps])
        
        fig, ax = plt.subplots()
        ax.plot(dts.detach().cpu().numpy())
        return {"figures/timesteps": [wandb.Image(fig)]}
    else:
        return {}


def init_dt(cfg, device=None):
    """Initialize dt parameters"""
    if cfg.per_step_dt:
        dt_schedule = cfg.sampler.dt_schedule
        steps = torch.arange(cfg.alg.actor.diff_steps, dtype=torch.float32, device=device)
        if dt_schedule is not None:
            dt_values = cfg.dt * dt_schedule(steps)
        else:
            dt_values = cfg.dt * torch.ones_like(steps)
        return inverse_softplus(dt_values)
    else:
        return torch.ones(1, device=device) * inverse_softplus(torch.tensor(cfg.dt, device=device))

