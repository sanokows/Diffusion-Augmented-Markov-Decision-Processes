"""Helpers for diffusion-timestep minibatch sampling with optional importance correction.

Two-stage view used in this file:
1) sample an original-state index uniformly,
2) sample a diffusion step t from q(t), biased toward low-noise steps.

With T diffusion steps:
  target distribution over (state, t): p = 1 / T
  proposal over t: q = q(t)
  IS ratio: w(t) = p / q = 1 / (T * q(t))

Rows are still sampled from flattened data. To implement the two-stage draw using
flattened row indices i, we use:
  q(i) = q(t_i) / n_{t_i},
where n_t is the number of rows at step t.

Sampling modes:
- "power":   q(t) ∝ ((t+1)/T)^exponent
             exponent=1.0  -> linear in normalized timestep
             exponent=2.0  -> quadratic
             exponent<1.0  -> flatter / less aggressive bias
- "song":    q(t) ∝ beta(t)^2 / alpha(t)^2, with
             alpha(t) = 1 - exp(- cumulative_beta(t))
  In this repository, beta is approximated from the rollout diffusion coefficient:
    beta_sample = eta
  and then averaged per diffusion step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def prepare_diffusion_importance_sampling(
    step_indices: jax.Array,
    diff_steps: int,
    *,
    sampling_mode: str = "power",
    exponent: float,
    beta_per_sample: jax.Array | None = None,
    song_reverse_time: bool = True,
    min_step_prob: float = 0.0,
    importance_clip: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return per-row proposal probs, IS ratios, and step-level proposal probs."""
    step_indices = jnp.asarray(step_indices, dtype=jnp.int32).reshape(-1)
    if diff_steps <= 0:
        raise ValueError("diff_steps must be positive.")
    step_indices = jnp.clip(step_indices, 0, diff_steps - 1)

    step_counts = jnp.bincount(step_indices, length=diff_steps).astype(jnp.float32)
    present = step_counts > 0
    eps = jnp.asarray(1e-8, dtype=jnp.float32)

    mode = str(sampling_mode).lower()
    if mode == "power":
        step_axis = jnp.arange(diff_steps, dtype=jnp.float32)
        # Higher diffusion step means lower noise in this codebase; exponent > 0
        # therefore oversamples lower-noise steps.
        step_weights = ((step_axis + 1.0) / float(diff_steps)) ** jnp.asarray(
            exponent, dtype=jnp.float32
        )
        step_probs = step_weights / jnp.maximum(jnp.sum(step_weights), eps)
    elif mode == "song":
        if beta_per_sample is None:
            raise ValueError(
                "beta_per_sample must be provided when sampling_mode='song'."
            )
        beta_per_sample = jnp.asarray(beta_per_sample, dtype=jnp.float32).reshape(-1)
        beta_sums = jnp.bincount(
            step_indices,
            weights=beta_per_sample,
            length=diff_steps,
        ).astype(jnp.float32)
        beta_step = beta_sums / jnp.maximum(step_counts, 1.0)
        beta_step = jnp.where(present, jnp.maximum(beta_step, eps), 0.0)

        if song_reverse_time:
            # In this codebase, larger diffusion step index means lower noise.
            # Reversed cumulative beta aligns Song-style weighting with this ordering.
            cumulative_beta = jnp.cumsum(beta_step[::-1])[::-1]
        else:
            cumulative_beta = jnp.cumsum(beta_step)
        alpha_step = 1.0 - jnp.exp(-cumulative_beta)
        step_weights = jnp.where(
            present,
            (beta_step**2) / jnp.maximum(alpha_step**2, eps),
            0.0,
        )
        step_probs = step_weights / jnp.maximum(jnp.sum(step_weights), eps)
    else:
        raise ValueError(
            f"Unknown sampling_mode '{sampling_mode}'. Expected 'power' or 'song'."
        )

    step_probs = jnp.where(present, step_probs, 0.0)
    step_probs = step_probs / jnp.maximum(jnp.sum(step_probs), eps)

    if min_step_prob > 0.0:
        step_probs = jnp.maximum(step_probs, jnp.asarray(min_step_prob, dtype=jnp.float32))
        step_probs = jnp.where(present, step_probs, 0.0)
        step_probs = step_probs / jnp.maximum(jnp.sum(step_probs), eps)

    per_sample_probs = step_probs[step_indices] / jnp.maximum(step_counts[step_indices], 1.0)
    step_prob_for_sample = step_probs[step_indices]
    importance_ratio = 1.0 / jnp.maximum(
        jnp.asarray(diff_steps, dtype=jnp.float32) * step_prob_for_sample,
        eps,
    )

    if importance_clip is not None and importance_clip > 0.0:
        importance_ratio = jnp.minimum(
            importance_ratio,
            jnp.asarray(importance_clip, dtype=jnp.float32),
        )

    return per_sample_probs, importance_ratio, step_probs


def sample_minibatch_indices(
    epoch_key: jax.Array,
    *,
    total_size: int,
    num_minibatches: int,
    mini_batch_size: int,
    use_importance_sampling: bool,
    per_sample_probs: jax.Array | None = None,
    per_sample_importance_ratio: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample/permute indices and return reshaped minibatch indices + IS ratios + keys."""
    if use_importance_sampling:
        sample_key, minibatch_key = jax.random.split(epoch_key)
        if per_sample_probs is None or per_sample_importance_ratio is None:
            raise ValueError(
                "per_sample_probs and per_sample_importance_ratio are required when "
                "use_importance_sampling=True."
            )
        indices = jax.random.choice(
            sample_key,
            total_size,
            shape=(total_size,),
            replace=True,
            p=per_sample_probs,
        )
        sampled_importance_ratio = jnp.take(per_sample_importance_ratio, indices, axis=0)
        minibatch_keys = jax.random.split(minibatch_key, num_minibatches)
    else:
        # Keep legacy behavior exactly for clean old/new comparisons.
        indices = jax.random.permutation(epoch_key, total_size)
        sampled_importance_ratio = jnp.ones((total_size,), dtype=jnp.float32)
        minibatch_keys = jax.random.split(epoch_key, num_minibatches)

    minibatch_idxs = indices.reshape((num_minibatches, mini_batch_size))
    minibatch_importance_ratio = sampled_importance_ratio.reshape(
        (num_minibatches, mini_batch_size)
    )
    return minibatch_idxs, minibatch_importance_ratio, minibatch_keys
