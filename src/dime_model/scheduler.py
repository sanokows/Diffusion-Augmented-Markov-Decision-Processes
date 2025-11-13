from typing import Tuple, Callable

import jax.numpy as jnp


def get_linear_schedule(total_steps, min=0.01):
    def linear_noise_schedule(step):
        t = (total_steps - step) / total_steps
        return (1. - t) * min + t

    return linear_noise_schedule


def get_cosine_schedule(total_steps, scale, min=0.01, s=0.008, pow=2):
    def cosine_schedule(step):
        t = (total_steps - step) / total_steps
        offset = 1 + s
        return ((1. - min) * jnp.cos(0.5 * jnp.pi * (offset - t) / offset) ** pow + min) * scale

    return cosine_schedule, None, None


def create_cos_square_scheduler(total_steps, C_start, C_end, T):
    a = (C_start - C_end)
    b = jnp.pi / (2 * T)
    c = C_end

    # def scheduler(step):
    #     """
    #     Cosine-squared scheduler that decreases smoothly from C_start to C_end
    #     Follows the shape: C_end + (C_start - C_end) * cos(π * t / T)^2
    #     """
    #     offset = 1 + 0.008
    #     t = (total_steps - step) / total_steps
    #     return a * (jnp.cos(b * (offset - t) / offset) ** 2) + c

    def scheduler(step):
        """
        Cosine-squared scheduler that decreases smoothly from C_start to C_end
        Follows the shape: C_end + (C_start - C_end) * cos(π * t / T)^2
        """
        t = step / total_steps
        # return (a * (jnp.cos(b * t) ** 2) + c)**2
        return a * (jnp.cos(b * t) ** 2) + c

    def forward_integrated_scheduler(step):
        """
        Compute the forward integral of the scheduler from 0 to t.
        This is the analytical integral of the linear scheduler from 0 to t.
        """
        t = step / total_steps
        return (a * jnp.sin(2 * b * t) / (4 * b)) + (a * t / 2) + (c * t)

    def backward_integrated_scheduler(step):
        """
        Compute the backward integral of the scheduler from s to T.
        This is the analytical integral of the linear scheduler from s to T.
        """
        s = step / total_steps
        return -(a * (jnp.sin(2 * b * s)) / (4 * b)) + (a * (T - s) / 2) + (c * (T - s))

    return scheduler, forward_integrated_scheduler, backward_integrated_scheduler


def get_constant_schedule():
    def constant_schedule(step):
        return jnp.array(1.)

    return constant_schedule


def get_geometric_schedule(total_steps, sigma_min: float, sigma_max: float) -> Tuple[
    Callable, Callable, Callable, Callable]:
    """
    Create a geometric noise scheduler following your provided formula (with correction).

    The formula appears to be: σ(t) = σ_min * (σ_max/σ_min)^(1-t/T) * sqrt(2 * log(σ_max/σ_min))

    Args:
        sigma_min (float): Minimum noise level
        sigma_max (float): Maximum noise level
        T (float): Total time duration

    Returns:
        Tuple of four functions:
        1. Scheduler function that returns the noise level at time t
        2. Forward integrated scheduler function that returns ∫_0^s scheduler(t) dt
        3. Delta integrated scheduler function that returns ∫_t_prev^t scheduler(t) dt
        4. Backward integrated scheduler function that returns ∫_s^T scheduler(t) dt
    """
    # Precompute constants
    ratio = sigma_max / sigma_min
    inv_ratio = sigma_min / sigma_max
    log_ratio = jnp.log(ratio)
    sqrt_factor = jnp.sqrt(2.0 * log_ratio)

    def scheduler(step):
        """Alternative geometric scheduler with sqrt factor"""
        t = step / total_steps
        return sigma_min * jnp.power(ratio, 1.0 - t) * sqrt_factor

    def forward_integrated_scheduler(step):
        """
        Compute the forward integral of the scheduler from 0 to t.
        This is the analytical integral of the linear scheduler from 0 to t.
        """
        t = step / total_steps
        return sigma_max ** 2 * (1 - inv_ratio ** (2 * t))

    return scheduler, None, None


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n_steps = 32
    cos_old = get_cosine_schedule(n_steps, 1.5, min=0.01)[0]
    cos_old_vals = [cos_old(step) for step in range(n_steps) ]

    cos_new = create_cos_square_scheduler(n_steps, 1.5, 0.01, 1.)[0]
    cos_new_vals = [cos_new(step) for step in range(n_steps) ]

    geometric = get_geometric_schedule(n_steps, 0.01, 1)[0]
    geometric_vals = [geometric(step) for step in range(n_steps) ]
    plt.plot(cos_old_vals, label='old cos')
    plt.plot(cos_new_vals, label='new cos')
    plt.plot(geometric_vals, label='geometric')
    plt.legend()
    plt.show()


