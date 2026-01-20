### AcrobotSwingup DMERL
python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingup hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=2e-3 hyperparameters.vmin=-50 hyperparameters.vmax=150 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=2e-3 hyperparameters.lagrangian_lr=2e-3 hyperparameters.ent_target_mult=3 hyperparameters.num_bins=201 hyperparameters.gamma=0.9992 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.use_friction_mlp=false hyperparameters.hl_gauss=true hyperparameters.ent_start=0.01 hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl hyperparameters.train_mode=WPO

### AcrobotSwingup DMERL
python -m src.jaxrl.reppo_DMERL_new env.name=PendulumSwingup hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=2e-3 hyperparameters.vmin=-50 hyperparameters.vmax=150 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=2e-3 hyperparameters.lagrangian_lr=2e-3 hyperparameters.ent_target_mult=3 hyperparameters.num_bins=201 hyperparameters.gamma=0.9992 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.use_friction_mlp=false hyperparameters.hl_gauss=true hyperparameters.ent_start=0.01 hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl hyperparameters.train_mode=WPO project_suffix=_WPO_test


### new 16.01
python -m src.jaxrl.reppo_DMERL_new env.name=WalkerRun hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.lr=1e-3 hyperparameters.temperature_lr=3e-4 hyperparameters.lagrangian_lr=3e-4 hyperparameters.ent_target_mult=2.5 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_dmc  hyperparameters.train_mode=WPO experiment_overrides=mjx_dmc_large_data_dmerl

### HopperHop
python -m src.jaxrl.reppo_DMERL_new env.name=HopperHop hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_WPO_hard_envs.yaml seed=0

python -m src.jaxrl.reppo_DMERL_new env.name=HopperHop hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_WPO_hard_envs.yaml seed=1

python -m src.jaxrl.reppo_DMERL_new env.name=HopperHop hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_WPO_hard_envs.yaml seed=2

### AcrobotSwingupSparse
python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingupSparse hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_WPO_hard_envs.yaml seed=0

python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingupSparse hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_WPO_hard_envs.yaml seed=1

python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingupSparse hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_WPO_hard_envs.yaml seed=2