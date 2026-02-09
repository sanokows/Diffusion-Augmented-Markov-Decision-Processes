# JAX + Optax + Flax: fit q(y|x)=N(y; x+mean_offset(x), sigma(x)) to a fixed 1D GMM f(y)
# minimizing KL(q(.|x) || f(.)) using the score-function (log-derivative) trick.
#
# Flax MLP: 2 hidden layers, ReLU activations.

import jax
import jax.numpy as jnp
import optax
from jax import random, lax
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
import argparse
from functools import partial
from flax import linen as nn
import wandb

# ----------------------------
# Target: fixed 3-component GMM f(y)
# ----------------------------
gmm_weights = jnp.array([0.35, 0.45, 0.20])
gmm_means   = jnp.array([-2.0, 0.5, 2.5])
gmm_stds    = jnp.array([0.4, 0.9, 0.25])

gmm_weights = gmm_weights / jnp.sum(gmm_weights)
log_w = jnp.log(gmm_weights + 1e-30)

LOG_2PI = jnp.log(2.0 * jnp.pi)
x_min = -6
x_max = 6

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_fourier", default=True, action="store_true", help="Use Fourier features for x encoding.")
    parser.add_argument("--num_fourier", type=int, default=64, help="Number of Fourier frequencies.")
    parser.add_argument("--fourier_scale", type=float, default=2.0, help="Scale for Fourier frequencies.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of residual blocks.")
    return parser.parse_args()

args = parse_args()

def normal_logpdf(y, mu, sigma):
    sigma = jnp.maximum(sigma, 1e-8)
    z = (y - mu) / sigma
    return -0.5 * (z * z + LOG_2PI) - jnp.log(sigma)

def gmm_logpdf(y):
    y_ = y[..., None]  # (..., 1)
    comp_logpdf = normal_logpdf(y_, gmm_means, gmm_stds)  # (..., K)
    return logsumexp(log_w + comp_logpdf, axis=-1)        # (...)

def gmm_pdf(y):
    return jnp.exp(gmm_logpdf(y))

def unbiased_kl_estimate(logq, logf, r_clip_max=1000.0):
    r = jnp.clip(jnp.exp(logf - logq), a_min=0.0, a_max=r_clip_max)
    return jnp.mean((r - 1.0) - (logf - logq))

def slope_mse(flax_params, x_batch):
    mean_offset, sigma = model.apply({'params': flax_params}, x_batch, deterministic=True)
    mu = x_batch + mean_offset
    q_x = jnp.exp(normal_logpdf(x_batch, mu, sigma))
    q_slope = q_x * (-(x_batch - mu) / (sigma * sigma))

    comp_pdf = jnp.exp(normal_logpdf(x_batch[:, None], gmm_means, gmm_stds))
    f_slope = jnp.sum(gmm_weights * comp_pdf * (-(x_batch[:, None] - gmm_means) / (gmm_stds ** 2)), axis=-1)
    diff = q_slope - f_slope
    return jnp.mean(diff ** 2), jnp.mean(jnp.abs(diff))

# ----------------------------
# Flax network: ResNet MLP with adjustable depth
# ----------------------------
class MLPMeanSigma(nn.Module):
    hidden_dim: int = 124
    dropout_rate: float = 0.1
    num_layers: int = 2
    use_fourier: bool = False
    num_fourier: int = 8
    fourier_scale: float = 1.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # x: (B,) or (B,1)
        if x.ndim == 1:
            x = x[:, None]  # (B,1)

        if self.use_fourier:
            freqs = (2.0 ** jnp.arange(self.num_fourier, dtype=x.dtype)) * self.fourier_scale
            x_proj = x * freqs[None, :]
            fourier = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
            x = jnp.concatenate([x, fourier], axis=-1)

        h = nn.Dense(self.hidden_dim)(x)
        h = nn.relu(h)
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
        for _ in range(self.num_layers):
            h_in = h
            h = nn.Dense(self.hidden_dim)(h)
            h = nn.relu(h)
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
            h = nn.Dense(self.hidden_dim)(h)
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=deterministic)
            h = nn.relu(h + h_in)
        out = nn.Dense(2)(h)  # (B,2)

        mean_offset = out[:, 0]
        log_sigma_raw = out[:, 1]
        sigma = nn.softplus(log_sigma_raw) + 1e-8
        return mean_offset, sigma

# ----------------------------
# KL + score-function surrogate
# ----------------------------
def kl_and_surrogate(flax_params, key, x_batch, num_eps):
    """
    KL(q(.|x)||f) = E_q[log q - log f]
    Score-function gradient:
      ∇ KL = E_q[ ( (log q - log f) + 1 ) ∇ log q ]
    Implement by surrogate:
      surrogate = E_q[ stopgrad(A_centered) * logq ]
    with A = (logq - logf + 1) and a baseline to reduce variance.
    """
    B = x_batch.shape[0]

    key_eps, key_drop = random.split(key, 2)
    mean_offset, sigma = model.apply(
        {'params': flax_params},
        x_batch,
        deterministic=False,
        rngs={'dropout': key_drop},
    )  # (B,), (B,)
    mu = x_batch + mean_offset

    eps = random.normal(key_eps, (B, num_eps))           # requires num_eps static under jit
    y = mu[:, None] + sigma[:, None] * eps

    y_stop = lax.stop_gradient(y)

    lam = 0.5
    logq = normal_logpdf(y_stop, mu[:, None], sigma[:, None])
    log_base = normal_logpdf(y_stop, x_batch[:, None], sigma[:, None])
    logf = gmm_logpdf(y_stop) + lam*( log_base - logq )

    g = logq - logf
    kl_est = jnp.mean(g)#unbiased_kl_estimate(logq, logf)

    A = g
    baseline = jnp.mean(A)
    A_centered = A - lax.stop_gradient(baseline)

    surrogate = jnp.mean(lax.stop_gradient(A_centered) * logq)
    return surrogate, kl_est

# ----------------------------
# Training
# ----------------------------
key = random.PRNGKey(0)
model = MLPMeanSigma(
    hidden_dim=124,
    dropout_rate=0.,
    num_layers=args.num_layers,
    use_fourier=args.use_fourier,
    num_fourier=args.num_fourier,
    fourier_scale=args.fourier_scale,
)

# init flax params
key, kinit = random.split(key)
dummy_x = jnp.zeros((1,), dtype=jnp.float32)
flax_params = model.init(kinit, dummy_x)['params']

lr = 1e-3
optimizer = optax.adam(lr)
opt_state = optimizer.init(flax_params)

# IMPORTANT: num_eps is arg index 4 -> make it static
@partial(jax.jit, static_argnums=(4,))
def train_step(flax_params, opt_state, key, x_batch, num_eps):
    def loss_fn(p):
        surrogate, kl = kl_and_surrogate(p, key, x_batch, num_eps)
        return surrogate, kl

    (surrogate, kl), grads = jax.value_and_grad(loss_fn, has_aux=True)(flax_params)
    updates, opt_state = optimizer.update(grads, opt_state, flax_params)
    flax_params = optax.apply_updates(flax_params, updates)
    return flax_params, opt_state, kl

steps = int(1e7)
batch_size = 128#256
num_eps = 34#64  # keep fixed (static) during training
print_every = 1000
fig_log_every = print_every

wandb.init(
    project="zero_order",
    config={
        "steps": steps,
        "batch_size": batch_size,
        "num_eps": num_eps,
        "lr": lr,
        "print_every": print_every,
        "fig_log_every": fig_log_every,
        "num_layers": args.num_layers,
        "use_fourier": args.use_fourier,
        "num_fourier": args.num_fourier,
        "fourier_scale": args.fourier_scale,
    },
)

def make_figure(flax_params, key, num_x=5, kl_samples=4096):
    x_test = random.uniform(key, (num_x,), minval=x_min, maxval=x_max)
    x_test = jnp.sort(x_test)

    ygrid = jnp.linspace(x_min, x_max, 2000)
    f_y = gmm_pdf(ygrid)

    mean_offset_test, sigma_test = model.apply({'params': flax_params}, x_test, deterministic=True)
    mu_test = x_test + mean_offset_test

    fig, axs = plt.subplots(len(x_test), 1, figsize=(10, 2.6 * len(x_test)), sharex=True)
    if len(x_test) == 1:
        axs = [axs]

    key, kkl = random.split(key)
    kl_keys = random.split(kkl, len(x_test))

    for i, (x0, mu0, s0, k) in enumerate(zip(x_test, mu_test, sigma_test, kl_keys)):
        q_y = jnp.exp(normal_logpdf(ygrid, mu0, s0))
        eps = random.normal(k, (kl_samples,))
        y_samp = mu0 + s0 * eps
        logq = normal_logpdf(y_samp, mu0, s0)
        logf = gmm_logpdf(y_samp)
        kl_est = unbiased_kl_estimate(logq, logf)

        axs[i].plot(ygrid, f_y, lw=2, label="GMM f(y)")
        axs[i].plot(ygrid, q_y, lw=2, label="learned Gaussian q(y|x)")
        axs[i].axvline(float(x0), lw=1.0, alpha=0.6, label="x")
        axs[i].axvline(float(mu0), lw=1.0, alpha=0.6, linestyle="--", label="x+mean_offset(x)")
        axs[i].set_title(
            f"x={float(x0):.3f} | mean_offset={float(mu0-x0):.3f} | sigma={float(s0):.3f} | KL(q||f)≈{float(kl_est):.4f}"
        )
        axs[i].set_ylabel("density")
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc="upper right")

    axs[-1].set_xlabel("y")
    axs[-1].set_xlim(x_min, x_max)
    plt.tight_layout()
    return fig

def make_slope_figure(flax_params, key, num_x=12):
    x_test = random.uniform(key, (num_x,), minval=x_min, maxval=x_max)
    x_test = jnp.sort(x_test)

    ygrid = jnp.linspace(x_min, x_max, 2000)
    f_y = gmm_pdf(ygrid)

    mean_offset_test, sigma_test = model.apply({'params': flax_params}, x_test, deterministic=True)
    mu_test = x_test + mean_offset_test

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(ygrid, f_y, lw=2.5, label="GMM f(y)")
    f_min = jnp.min(f_y)
    f_max = jnp.max(f_y)

    colors = plt.cm.viridis(jnp.linspace(0.1, 0.9, len(x_test)))
    for i, (x0, mu0, s0, c) in enumerate(zip(x_test, mu_test, sigma_test, colors)):
        f_x = gmm_pdf(x0)
        comp_pdf = jnp.exp(normal_logpdf(x0, gmm_means, gmm_stds))
        f_slope = jnp.sum(gmm_weights * comp_pdf * (-(x0 - gmm_means) / (gmm_stds ** 2)))
        logq_x = normal_logpdf(x0, mu0, s0)
        q_x = jnp.exp(logq_x)
        slope = q_x * (-(x0 - mu0) / (s0 * s0))
        y_line = f_x + slope * (ygrid - x0)
        y_line = jnp.where((y_line >= f_min) & (y_line <= f_max), y_line, jnp.nan)
        ax.plot(ygrid, y_line, color=c, alpha=0.8, lw=1.5, linestyle=":")
        y_line_f = f_x + f_slope * (ygrid - x0)
        y_line_f = jnp.where((y_line_f >= f_min) & (y_line_f <= f_max), y_line_f, jnp.nan)
        ax.plot(
            ygrid,
            y_line_f,
            color="red",
            alpha=0.7,
            lw=1.5,
            linestyle="--",
            label="f'(x) tangent" if i == 0 else None,
        )
        ax.scatter([float(x0)], [float(f_x)], color=c, s=24, zorder=3)

    ax.set_title("f(y) with tangent lines using q'(x)")
    ax.set_xlabel("y")
    ax.set_ylabel("density / line value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig

def make_mean_sigma_figure(flax_params, key, num_points=400):
    x_grid = jnp.linspace(x_min, x_max, num_points)
    mean_offset, sigma = model.apply({'params': flax_params}, x_grid, deterministic=True)
    mu = x_grid + mean_offset

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(x_grid, mu, lw=2.0, label="mean(x)")
    ax[0].plot(x_grid, x_grid, lw=1.5, linestyle="--", alpha=0.7, label="y=x")
    ax[0].set_ylabel("mean")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="upper right")

    ax[1].plot(x_grid, sigma, lw=2.0, label="sigma(x)")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("sigma")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc="upper right")

    plt.tight_layout()
    return fig

def make_kl_over_x_figure(flax_params, key, num_points=200, kl_samples=1024):
    x_grid = jnp.linspace(x_min, x_max, num_points)
    mean_offset, sigma = model.apply({'params': flax_params}, x_grid, deterministic=True)
    mu = x_grid + mean_offset

    eps = random.normal(key, (num_points, kl_samples))
    y_samp = mu[:, None] + sigma[:, None] * eps
    logq = normal_logpdf(y_samp, mu[:, None], sigma[:, None])
    logf = gmm_logpdf(y_samp)
    kl_vals = jnp.mean((jnp.clip(jnp.exp(logf - logq), a_min=0.0, a_max=1000.0) - 1.0) - (logf - logq), axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    ax.plot(x_grid, kl_vals, lw=2.0, label="KL(q||f) at x")
    ax.set_xlabel("x")
    ax.set_ylabel("KL estimate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig

def make_integrated_qslope_figure(flax_params, num_points=400):
    x_grid = jnp.linspace(x_min, x_max, num_points)
    mean_offset, sigma = model.apply({'params': flax_params}, x_grid, deterministic=True)
    mu = x_grid + mean_offset
    q_x = jnp.exp(normal_logpdf(x_grid, mu, sigma))
    q_slope = q_x * (-(x_grid - mu) / (sigma * sigma))

    dx = x_grid[1] - x_grid[0]
    trap = 0.5 * (q_slope[:-1] + q_slope[1:]) * dx
    q_int = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(trap)])

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    ax.plot(x_grid, q_int, lw=2.0, label="∫ q'(x) dx (up to const)")
    ax.set_xlabel("x")
    ax.set_ylabel("integral value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig

for step in range(1, steps + 1):
    key, kx, kstep = random.split(key, 3)
    x_batch = random.uniform(kx, (batch_size,), minval=x_min, maxval=x_max)

    slope_mse_val, slope_l1_val = slope_mse(flax_params, x_batch)
    flax_params, opt_state, kl_val = train_step(flax_params, opt_state, kstep, x_batch, num_eps)

    wandb.log(
        {
            "kl_estimate": float(kl_val),
            "slope_mse": float(slope_mse_val),
            "slope_l1": float(slope_l1_val),
        },
        step=step,
    )

    if step % fig_log_every == 0 or step == 1:
        key, kfig, kslope, kmean, kkl = random.split(key, 5)
        fig = make_figure(flax_params, kfig)
        wandb.log({"comparison": wandb.Image(fig)}, step=step)
        plt.close(fig)
        fig = make_slope_figure(flax_params, kslope)
        wandb.log({"slope_comparison": wandb.Image(fig)}, step=step)
        plt.close(fig)
        fig = make_mean_sigma_figure(flax_params, kmean)
        wandb.log({"mean_sigma": wandb.Image(fig)}, step=step)
        plt.close(fig)
        fig = make_kl_over_x_figure(flax_params, kkl)
        wandb.log({"kl_over_x": wandb.Image(fig)}, step=step)
        plt.close(fig)
        fig = make_integrated_qslope_figure(flax_params)
        wandb.log({"integrated_q_slope": wandb.Image(fig)}, step=step)
        plt.close(fig)

    if step % print_every == 0 or step == 1:
        print(f"step {step:5d} | KL estimate: {float(kl_val):.5f}")

# ----------------------------
# Plot learned Gaussians vs GMM at a few x's
# ----------------------------
key, kplot = random.split(key)
fig = make_figure(flax_params, kplot)
plt.show()
