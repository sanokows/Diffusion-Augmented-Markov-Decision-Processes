
### works??
exec python -m src.jaxrl.reppo_DMERL_new env.name=TurningDoubleWellEnv hyperparameters.num_eval=50 hyperparameters.total_time_steps=10000000 hyperparameters.diffusion.diff_steps=8 env=mjx_double_well experiment_overrides=dmerl_WPO/mjx_double_well_DMERL_WPO hyperparameters.kl_bound=0.12 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.ent_target_mult=6 hyperparameters.max_grad_norm=0.5 hyperparameters.hl_gauss=false hyperparameters.reverse_kl=true hyperparameters.stop_grad_entropy=true hyperparameters.actor_kl_clip_mode="full"

# New temperature mode for WPO:
# - keeps temperature fixed at `hyperparameters.ent_start`
# - forces `hyperparameters.stop_grad_entropy=false`
# - uses a separate entropy lagrangian for target entropy loss
# enable with:
#   hyperparameters.new_temp_mode=true


### works??
exec python -m src.jaxrl.reppo_DMERL_new env.name=WalkerRun hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl_WPO/mjx_dmc_large_data_dmerl_WPO_linear_schedule 

exec python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl_WPO/mjx_dmc_large_data_dmerl_WPO_linear_schedule 

exec python -m src.jaxrl.reppo_DMERL_new env.name=HopperHop hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl_WPO/mjx_dmc_large_data_dmerl_WPO_linear_schedule 

### try smaller lr or smaller ent_factor
salloc -A EU-25-100 -p qgpu --time=11:00:30 

  #ent_target_mult: 8.

  entropy_lagrangian_start: 0.01
  
  #lr: 1e-3
  #temperature_lr: 3e-4
  #lagrangian_lr: 3e-4

exec python -m src.jaxrl.reppo_DMERL_new env.name=FingerSpin wandb.project_suffix=_FR_WPO_02_05_test hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.train_mode=WPO hyperparameters.ent_target_mult=10 env=mjx_dmc num_trials=1 seed=0 experiment_overrides=dmerl_WPO/mjx_dmc_large_data_dmerl_WPO_linear_schedule_test

exec python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingup wandb.project_suffix=_FR_WPO_02_05_test hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.train_mode=WPO hyperparameters.ent_target_mult=10 env=mjx_dmc num_trials=1 seed=0 experiment_overrides=dmerl_WPO/mjx_dmc_large_data_dmerl_WPO_linear_schedule_test

exec python -m src.jaxrl.reppo_DMERL_new env.name=FingerSpin wandb.project_suffix=_FR_WPO_02_05_test hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.train_mode=WPO hyperparameters.ent_target_mult=9 env=mjx_dmc num_trials=1 seed=0 experiment_overrides=dmerl_WPO/mjx_dmc_large_data_dmerl_WPO_linear_schedule_test hyperparameters.lr=1e-3 hyperparameters.temperature_lr=6e-4 hyperparameters.lagrangian_lr=6e-4 hyperparameters.kl_bound=0.08

exec python -m src.jaxrl.reppo_DMERL_new env.name=FingerSpin wandb.project_suffix=_FR_WPO_02_05_test hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.train_mode=WPO hyperparameters.ent_target_mult=9 env=mjx_dmc num_trials=1 seed=0 experiment_overrides=dmerl_WPO/mjx_dmc_large_data_dmerl_WPO_linear_schedule_test hyperparameters.lr=1e-3 hyperparameters.temperature_lr=6e-4 hyperparameters.lagrangian_lr=6e-4 hyperparameters.kl_bound=0.14

exec python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingup wandb.project_suffix=_FR_WPO_02_05_test hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.train_mode=WPO hyperparameters.ent_target_mult=9 env=mjx_dmc num_trials=1 seed=0 experiment_overrides=dmerl_WPO/mjx_dmc_large_data_dmerl_WPO_linear_schedule_test hyperparameters.lr=1e-3 hyperparameters.temperature_lr=6e-4 hyperparameters.lagrangian_lr=6e-4 hyperparameters.kl_bound=0.08


exec python -m src.jaxrl.reppo_DMERL_new \
  env.name=FingerSpin \
  wandb.project_suffix=_FR_WPO_02_05_test \
  hyperparameters.num_eval=50 \
  hyperparameters.total_time_steps=50000000 \
  hyperparameters.diffusion.diff_steps=8 \
  hyperparameters.train_mode=WPO \
  hyperparameters.new_temp_mode=true \
  hyperparameters.update_entropy_lagrangian=true \
  hyperparameters.use_temperature_decay=false \
  hyperparameters.stop_grad_entropy=true \
  hyperparameters.ent_target_mult=9 \
  env=mjx_dmc \
  num_trials=1 \
  seed=0 \
  experiment_overrides=dmerl_WPO/mjx_dmc_large_data_dmerl_WPO_linear_schedule_test \
  hyperparameters.lr=1e-3 \
  hyperparameters.temperature_lr=6e-4 \
  hyperparameters.lagrangian_lr=6e-4

  exec python -m src.jaxrl.reppo_DMERL_new \
  env.name=FingerSpin \
  wandb.project_suffix=_FR_WPO_02_05_test \
  hyperparameters.num_eval=50 \
  hyperparameters.total_time_steps=50000000 \
  hyperparameters.diffusion.diff_steps=8 \
  hyperparameters.train_mode=WPO \
  hyperparameters.new_temp_mode=true \
  hyperparameters.update_entropy_lagrangian=true \
  hyperparameters.use_temperature_decay=false \
  hyperparameters.stop_grad_entropy=false \
  hyperparameters.ent_target_mult=9 \
  env=mjx_dmc \
  num_trials=1 \
  seed=0 \
  experiment_overrides=dmerl_WPO/mjx_dmc_large_data_dmerl_WPO_linear_schedule_test \
  hyperparameters.lr=1e-3 \
  hyperparameters.temperature_lr=6e-5 \
  hyperparameters.lagrangian_lr=6e-5
