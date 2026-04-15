
### works??
exec python -m src.jaxrl.reppo_DMERL_new env.name=TurningDoubleWellEnv hyperparameters.num_eval=50 hyperparameters.total_time_steps=10000000 hyperparameters.diffusion.diff_steps=8 env=mjx_double_well experiment_overrides=dmerl_WPO/mjx_double_well_DMERL_WPO hyperparameters.kl_bound=0.12 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.ent_target_mult=6 hyperparameters.max_grad_norm=0.5 hyperparameters.hl_gauss=false hyperparameters.reverse_kl=true hyperparameters.stop_grad_entropy=true hyperparameters.actor_kl_clip_mode="full"

# New temperature mode for WPO:
# - keeps temperature fixed at `hyperparameters.ent_start`
# - forces `hyperparameters.stop_grad_entropy=false`
# - uses a separate entropy lagrangian for target entropy loss
# enable with:
#   hyperparameters.new_temp_mode=true


### works??
exec python -m src.jaxrl.reppo_DMERL_new env.name=WalkerRun hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl_WPO/mjx_dmc_large_data_dmerl_WPO_linear_schedule hyperparameters.kl_bound=0.12 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.ent_target_mult=4 hyperparameters.hl_gauss=true hyperparameters.new_temp_mode=true hyperparameters.ent_start=0.005 hyperparameters.entropy_lagrangian_start=0.01 hyperparameters.remove_fisher_precond=false hyperparameters.actor_kl_clip_mode="full"