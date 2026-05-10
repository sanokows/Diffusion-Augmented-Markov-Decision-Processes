# Architecture Notes: `reppo_PPO.py` vs `reppo.py`

This note summarizes the main model-architecture differences between:
- `src/jaxrl/reppo_PPO.py`
- `src/jaxrl/reppo.py` (using `src/networks/jax_models.py`)

## High-level summary

`reppo_PPO.py` uses a classic small PPO MLP (fixed shape), while `reppo.py` uses larger and more modular actor/critic networks with more architecture options.

## Side-by-side differences

| Aspect | `reppo_PPO.py` | `reppo.py` |
|---|---|---|
| Hidden width (neurons) | `hidden_dim=64` in actor + critic | Default `actor_hidden_dim=512`, `critic_hidden_dim=512` |
| Actor depth | 2 hidden layers (`Linear -> tanh -> Linear -> tanh -> Linear`) | Configurable FCNN, default `num_actor_layers=3` |
| Critic depth | 2 hidden layers, single scalar value head | Modular critic with encoder + value head + prediction head (default `2/2/2` layers) |
| Normalization in network | No LayerNorm in PPO network | LayerNorm used in FCNN blocks when `use_actor_norm/use_critic_norm=true` (default true) |
| Activation | `tanh` | `swish` (in FCNN blocks) |
| Actor output parameterization | Mean from MLP + global state-independent `log_std` parameter | Mean and log-std are both state-dependent (`action_dim*2` output split into mean/log_std) |
| Action distribution | `MultivariateNormalDiag` | `Normal` transformed with `Tanh` |
| Critic target/output style | Scalar value prediction | Usually categorical critic (`hl_gauss=true`, `num_bins=151`) in default `reppo.yaml`; can switch to scalar critic |
| Skip connections | None | Optional (`use_actor_skip`, `use_critic_skip`) |

## Important config note

`config/PPO_overrides/*.yaml` (for `reppo_PPO.py`) mainly changes training behavior (e.g., reward normalization, adaptive LR), not network shape.  
The PPO architecture remains the same unless `PPONetworks` code is changed directly.
