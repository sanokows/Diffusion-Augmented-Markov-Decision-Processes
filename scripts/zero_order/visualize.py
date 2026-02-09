# --- Colab-friendly interactive plotting with ipywidgets ---
# If widgets don't display, run this cell once:
# !pip -q install ipywidgets
from google.colab import output
output.enable_custom_widget_manager()

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# ----------------------------
# Gaussian + GMM utilities
# ----------------------------
def gaussian_pdf(z, mu, sigma):
    sigma = max(float(sigma), 1e-12)
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * ((z - mu) / sigma) ** 2)

def gmm_pdf(xgrid, weights, mus, sigmas):
    w = np.asarray(weights, dtype=float)
    m = np.asarray(mus, dtype=float)
    s = np.asarray(sigmas, dtype=float)

    wsum = w.sum()
    if wsum <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / wsum

    pdf = np.zeros_like(xgrid, dtype=float)
    for wi, mi, si in zip(w, m, s):
        pdf += wi * gaussian_pdf(xgrid, mi, si)
    return pdf

def kl_divergence_eps(p, q, eps_grid):
    """KL(P||Q) = ∫ p log(p/q) dε (numerical over eps_grid)."""
    tiny = 1e-300
    p = np.clip(p, tiny, None)
    q = np.clip(q, tiny, None)
    return np.trapz(p * (np.log(p) - np.log(q)), eps_grid)

def make_q_eps_from_f(x, sigma, eps_grid, weights, mus, sigmas):
    """
    Q(ε) induced by Y = x + εσ with Y~f:
      q(ε) = σ f(x + εσ)
    Renormalize on finite eps_grid for numerical stability.
    Returns:
      q_eps (density over ε, renorm on eps_grid),
      f_shift_unscaled = f(x+εσ) (not a density over ε).
    """
    f_shift_unscaled = gmm_pdf(x + eps_grid * sigma, weights, mus, sigmas)  # f(x+εσ)
    q = sigma * f_shift_unscaled
    Z = np.trapz(q, eps_grid)
    if Z <= 0 or not np.isfinite(Z):
        q = np.ones_like(q) / (eps_grid[-1] - eps_grid[0])
    else:
        q = q / Z
    return q, f_shift_unscaled

# ----------------------------
# Maxima helpers (for vertical lines)
# ----------------------------
def local_maxima_positions(x, y):
    """
    Strict local maxima indices i where y[i] > y[i-1] and y[i] > y[i+1].
    Returns positions and indices.
    """
    y = np.asarray(y)
    idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    return np.asarray(x)[idx], idx

def prune_close_positions(xpts, y_at_xpts, min_sep=0.15):
    """If peaks are too close in x, keep only the highest one within min_sep."""
    xpts = np.asarray(xpts)
    y_at_xpts = np.asarray(y_at_xpts)
    if xpts.size == 0:
        return xpts

    order = np.argsort(-y_at_xpts)  # keep highest first
    keep = []
    for j in order:
        if all(abs(xpts[j] - xpts[k]) >= min_sep for k in keep):
            keep.append(j)
    keep = np.array(sorted(keep), dtype=int)
    return xpts[keep]

# ----------------------------
# Fixed mixture parameters (edit if desired)
# ----------------------------
weights = [0.35, 0.45, 0.20]
mus     = [-2.0,  0.5,  2.5]
sigmas  = [0.4,   0.9,  0.25]  # varying std devs

# Grids
xgrid   = np.linspace(-8, 8, 2500)   # for plotting f(x)
eps_grid = np.linspace(-6, 6, 2401)  # for ε integration
x_eval  = np.linspace(-5, 5, 250)    # for KL(x) curve

# For the 2nd plot x-range requirement: always show -8..8
# We'll plot vs y = x + εσ, and build y-grid in [-8,8].
ygrid = np.linspace(-8, 8, 2401)

# ----------------------------
# Widgets
# ----------------------------
x_slider     = widgets.FloatSlider(value=0.0, min=-5.0, max=5.0, step=0.05, description='x')
sigma_slider = widgets.FloatSlider(value=1.0, min=0.05, max=3.0, step=0.05, description='sigma')

kl_direction = widgets.ToggleButtons(
    options=[('KL(P||Q)', 'P||Q'), ('KL(Q||P)', 'Q||P')],
    value='P||Q',
    description='KL:'
)

# controls how aggressively we merge nearby maxima lines
min_sep_slider = widgets.FloatSlider(value=0.25, min=0.0, max=1.0, step=0.05, description='min sep')

out = widgets.Output()

def update_plot(change=None):
    with out:
        clear_output(wait=True)

        x = float(x_slider.value)
        sigma = float(sigma_slider.value)
        kl_dir = kl_direction.value
        min_sep = float(min_sep_slider.value)

        # --- (A) f(x): 3-component GMM ---
        fx = gmm_pdf(xgrid, weights, mus, sigmas)

        # --- (B) Over ε: P(ε) and Q(ε) use eps_grid ---
        # P(ε) = N(ε; x, σ) as requested; renormalize on finite eps_grid
        p_eps = gaussian_pdf(eps_grid, x, sigma)
        pZ = np.trapz(p_eps, eps_grid)
        p_eps = p_eps / pZ if (pZ > 0 and np.isfinite(pZ)) else np.ones_like(p_eps) / (eps_grid[-1] - eps_grid[0])

        # Q(ε) induced by f via y=x+εσ
        q_eps, f_shift_unscaled = make_q_eps_from_f(x, sigma, eps_grid, weights, mus, sigmas)

        # KL at current x
        kl_here = kl_divergence_eps(p_eps, q_eps, eps_grid) if kl_dir == 'P||Q' else kl_divergence_eps(q_eps, p_eps, eps_grid)

        # --- (C) KL over x (integrated over ε numerically) ---
        kls = np.zeros_like(x_eval)
        for i, xv in enumerate(x_eval):
            p = gaussian_pdf(eps_grid, xv, sigma)
            pZ = np.trapz(p, eps_grid)
            p = p / pZ if (pZ > 0 and np.isfinite(pZ)) else np.ones_like(p) / (eps_grid[-1] - eps_grid[0])

            q, _ = make_q_eps_from_f(xv, sigma, eps_grid, weights, mus, sigmas)

            kls[i] = kl_divergence_eps(p, q, eps_grid) if kl_dir == 'P||Q' else kl_divergence_eps(q, p, eps_grid)

        # ----------------------------
        # Vertical lines requested:
        # positions of the MAXIMA of f(x + εσ).
        #
        # On the KL-vs-x panel, the x-axis is x. The meaningful x-locations
        # you can draw as vertical lines are the peaks of f(x) (i.e., ε=0),
        # which correspond to peaks of the shifted function when ε=0.
        # ----------------------------
        f_on_xeval = gmm_pdf(x_eval, weights, mus, sigmas)  # f(x)
        xpeaks, idx_pk = local_maxima_positions(x_eval, f_on_xeval)
        if xpeaks.size > 0:
            xpeaks = prune_close_positions(xpeaks, f_on_xeval[idx_pk], min_sep=min_sep)

        # ----------------------------
        # Second plot x-range requirement (-8..8):
        # Plot over y in [-8,8], showing:
        #   - f(y) where y = x + εσ (but f depends only on y)
        #   - the induced y-density from ε~N(ε; x,σ): p_y(y) = (1/σ) N((y-x)/σ; x,σ)
        # This keeps the horizontal axis fixed to [-8,8].
        # ----------------------------
        f_y = gmm_pdf(ygrid, weights, mus, sigmas)

        # p_y(y) induced by ε ~ N(ε; x, σ) through y = x + σ ε
        # ε = (y - x)/σ, so p_y(y) = p_ε((y-x)/σ) * (1/σ)
        eps_from_y = (ygrid - x) / sigma
        p_y = gaussian_pdf(eps_from_y, x, sigma) * (1.0 / sigma)
        # renormalize p_y over the displayed y-grid (finite domain)
        p_yZ = np.trapz(p_y, ygrid)
        if p_yZ > 0 and np.isfinite(p_yZ):
            p_y = p_y / p_yZ

        # ----------------------------
        # Plot
        # ----------------------------
        fig, axs = plt.subplots(3, 1, figsize=(10, 11))
        plt.subplots_adjust(hspace=0.35)

        # (A) f(x)
        axs[0].plot(xgrid, fx, lw=2, label='f(x) (3-comp GMM)')
        axs[0].axvline(x, lw=1.5, alpha=0.7, label='current x')
        axs[0].set_title("1-D Gaussian Mixture density f(x)")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("density")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()

        # (B) fixed x-range -8..8 (plot over y)
        axs[1].plot(ygrid, f_y, lw=2, label=r"$f(y)$ where $y=x+\epsilon\sigma$")
        axs[1].plot(ygrid, p_y, lw=2, label=r"induced $p(y)$ from $\epsilon\sim\mathcal{N}(\epsilon;x,\sigma)$")
        axs[1].axvline(x, lw=1.2, alpha=0.6, label="current x")
        axs[1].set_xlim(-8, 8)
        axs[1].set_title("Fixed x-range [-8,8]: compare f(y) and induced p(y)")
        axs[1].set_xlabel("y ( = x + εσ )")
        axs[1].set_ylabel("value (renorm on shown range)")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()

        # (C) KL vs x with vertical lines at peaks of f(x)
        axs[2].plot(x_eval, kls, lw=2, label="KL(x)")
        axs[2].axvline(x, lw=1.2, alpha=0.7, label='current x')
        for xp in xpeaks:
            axs[2].axvline(xp, lw=1.5, alpha=0.8)

        axs[2].set_title(rf"KL over $\epsilon$ vs x (numeric), with $\sigma={sigma:.2f}$")
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("KL")
        axs[2].grid(True, alpha=0.3)

        if xpeaks.size > 0:
            axs[2].legend(title=f"peaks of f(x): {len(xpeaks)}", loc="best")
        else:
            axs[2].legend(loc="best")

        # Also show KL at current x as text
        axs[2].text(
            0.02, 0.95,
            f"{'KL(P||Q)' if kl_dir=='P||Q' else 'KL(Q||P)'} at current x: {kl_here:.4f}",
            transform=axs[2].transAxes, va='top'
        )

        plt.show()

# Initial draw + callbacks
update_plot()
x_slider.observe(update_plot, names='value')
sigma_slider.observe(update_plot, names='value')
kl_direction.observe(update_plot, names='value')
min_sep_slider.observe(update_plot, names='value')

display(widgets.VBox([
    widgets.HBox([x_slider, sigma_slider]),
    widgets.HBox([kl_direction, min_sep_slider]),
    out
]))
