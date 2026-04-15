
### CheetahRun reppo_dime smaller lr necessary
exec python -m src.jaxrl.reppo_dime \
        env.name=FishSwim \
        wandb.project_suffix="_FR_29_03" \
        hyperparameters.num_eval=50 \
        hyperparameters.total_time_steps=50000000 \
        hyperparameters.diffusion.diff_steps=8 \
        hyperparameters.kl_action_rep=4 \
        hyperparameters.reverse_kl=false \
        hyperparameters.actor_kl_clip_mode=clipped \
        hyperparameters.ent_start=0.01 \
        hyperparameters.vmin=-100 \
        hyperparameters.vmax=200 \
        hyperparameters.num_bins=301 \
        hyperparameters.diffusion.learn_friction=true \
        hyperparameters.diffusion.learn_dt=true \
        hyperparameters.diffusion.per_step_dt=true \
        hyperparameters.lr=3e-4 \
        hyperparameters.temperature_lagragian_lr=1e-4 \
        env=mjx_dmc \
        seed=0 \
        num_trials=1 \
        experiment_overrides=dime/mjx_dmc_large_data 

### todo test smaller vmin and vmax only two critic layers
exec python -m src.jaxrl.reppo_dime env.name=TurningDoubleWellEnv hyperparameters.num_eval=50 hyperparameters.total_time_steps=10000000 hyperparameters.diffusion.diff_steps=8 env=mjx_double_well experiment_overrides=dime/mjx_double_well_dime