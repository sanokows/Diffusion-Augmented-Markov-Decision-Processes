import jax.numpy as jnp


def _schedule_progress(step, total_steps):
    """Return clipped progress in [0, 1] over valid indices [0, total_steps - 1]."""
    step = jnp.asarray(step, dtype=jnp.float32)
    total_steps = jnp.asarray(total_steps, dtype=jnp.float32)
    denom = jnp.maximum(total_steps - 1.0, 1.0)
    return jnp.clip(step / denom, 0.0, 1.0)


def get_linear_schedule(total_steps, min=0.01, s=0.008, pow=2):
    def linear_noise_schedule(step):
        progress = _schedule_progress(step, total_steps)
        return 1.0 + (min - 1.0) * progress

    return linear_noise_schedule


def get_cosine_schedule(total_steps, min=0.01, s=0.008, pow=2):
    def cosine_schedule(step):
        progress = _schedule_progress(step, total_steps)
        t = 1.0 - progress
        offset = 1 + s
        return (1. - min) * jnp.cos(0.5 * jnp.pi * (offset - t) / offset) ** pow + min

    return cosine_schedule


def get_constant_schedule(total_steps, min=0.01, s=0.008, pow=2):
    def constant_schedule(step):
        return jnp.array(1.)

    return constant_schedule
