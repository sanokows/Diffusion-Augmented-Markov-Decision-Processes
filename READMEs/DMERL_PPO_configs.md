

exec python -m src.jaxrl.DA_MDP_PPO env.name=TurningDoubleWellEnv wandb.project_suffix=_FR_test_PPO hyperparameters.num_eval=50 hyperparameters.total_time_steps=10000000 hyperparameters.num_mini_batches=8 hyperparameters.lr=6e-4 hyperparameters.temperature_lr=1e-4 hyperparameters.entropy_coef=0.05 hyperparameters.update_entropy_lagrangian=false hyperparameters.num_envs=1024 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.use_categorical_value=true hyperparameters.normalize_advantages=false hyperparameters.ent_target_mult=8. hyperparameters.num_collection_step_factor=0.5 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.num_bins=301 hyperparameters.num_epochs=8 hyperparameters.diffusion.friction=0.25 hyperparameters.diffusion.init_std=3. env=mjx_double_well hyperparameters.diffusion.dt_schedule.min=0.01


exec python -m src.jaxrl.DA_MDP_PPO env.name=TurningDoubleWellEnv wandb.project_suffix=_FR_test_PPO hyperparameters.num_eval=50 hyperparameters.total_time_steps=10000000 hyperparameters.num_mini_batches=8 hyperparameters.lr=6e-4 hyperparameters.temperature_lr=1e-4 hyperparameters.entropy_coef=0.0000 hyperparameters.update_entropy_lagrangian=false hyperparameters.num_envs=1024 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.use_categorical_value=true hyperparameters.normalize_advantages=false hyperparameters.ent_target_mult=8. hyperparameters.num_collection_step_factor=0.5 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.num_bins=301 hyperparameters.num_epochs=8 hyperparameters.diffusion.friction=0.25 hyperparameters.diffusion.init_std=3. env=mjx_double_well hyperparameters.diffusion.dt_schedule.min=0.01

python -m src.jaxrl.DA_MDP_PPO env.name=PendulumSwingup overrides=default env=mjx_dmc overrides.hyperparameters.total_time_steps=25000000 wandb.project_suffix=_FR_13_04 overrides.hyperparameters.lr=1e-3 overrides.hyperparameters.entropy_coef=1e-5 overrides.hyperparameters.num_envs=4048 overrides.hyperparameters.temperature_lr=3e-4 overrides.hyperparameters.update_entropy_lagrangian=true overrides.hyperparameters.num_epochs=12 overrides.hyperparameters.ent_target_mult=6 seed=0 trials=1


python -m src.jaxrl.DA_MDP_PPO env.name=PendulumSwingup overrides=default env=mjx_dmc wandb.project_suffix=_FR_PPO_27_04_test trials=1 seed=0 hyperparameters.clip_ratio=0.1


## CheetahRun DiffPPO comparison (plain value)

### 1) Old baseline config (default) in plain-value mode
python -m src.jaxrl.DA_MDP_PPO \
  env=mjx_dmc \
  env.name=CheetahRun \
  overrides=default \
  hyperparameters.use_categorical_value=false \
  hyperparameters.hl_gauss=false

### 2) Old DPPO config in plain-value mode
python -m src.jaxrl.DA_MDP_PPO \
  env=mjx_dmc \
  env.name=CheetahRun \
  overrides=DA_MDP_PPO \
  hyperparameters.use_categorical_value=false \
  hyperparameters.hl_gauss=false

### 3) New features enabled (reward normalization + adaptive LR), plain-value preset
python -m src.jaxrl.DA_MDP_PPO \
  env=mjx_dmc \
  env.name=CheetahRun \
  overrides=default \
  features=adaptive_rewardnorm_adaptivelr_plain_value

### 3) New features enabled (reward normalization + adaptive LR), plain-value preset
python -m src.jaxrl.DA_MDP_PPO \
  env=mjx_dmc \
  env.name=CheetahRun \
  overrides=DA_MDP_PPO_ent_reg \
  features=adaptive_rewardnorm_adaptivelr_plain_value

### 3) New features enabled (reward normalization + adaptive LR), plain-value preset
python -m src.jaxrl.DA_MDP_PPO \
  env=mjx_dmc \
  env.name=PendulumSwingup \
  overrides=DA_MDP_PPO_ent_reg \
  features=adaptive_rewardnorm_adaptivelr_plain_value

### 4) DPPO mode + new features enabled (reward normalization + adaptive LR), plain-value preset
python -m src.jaxrl.DA_MDP_PPO \
  env=mjx_dmc \
  env.name=CheetahRun \
  overrides=DA_MDP_PPO \
  features=adaptive_rewardnorm_adaptivelr_plain_value

### Optional dry-run sanity checks (no training)
python -m src.jaxrl.DA_MDP_PPO \
  env=mjx_dmc \
  env.name=CheetahRun \
  overrides=default \
  hyperparameters.use_categorical_value=false \
  hyperparameters.hl_gauss=false \
  trials=0 \
  wandb.mode=disabled

python -m src.jaxrl.DA_MDP_PPO \
  env=mjx_dmc \
  env.name=CheetahRun \
  overrides=DA_MDP_PPO \
  hyperparameters.use_categorical_value=false \
  hyperparameters.hl_gauss=false \
  trials=0 \
  wandb.mode=disabled

python -m src.jaxrl.DA_MDP_PPO \
  env=mjx_dmc \
  env.name=CheetahRun \
  overrides=default \
  features=adaptive_rewardnorm_adaptivelr_plain_value \
  trials=0 \
  wandb.mode=disabled

python -m src.jaxrl.DA_MDP_PPO \
  env=mjx_dmc \
  env.name=CheetahRun \
  overrides=DA_MDP_PPO \
  features=adaptive_rewardnorm_adaptivelr_plain_value \
  trials=0 \
  wandb.mode=disabled
