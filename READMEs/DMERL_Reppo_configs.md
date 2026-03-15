### AcrobotSwingup DMERL
python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingup hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.lr=6e-4 hyperparameters.temperature_lr=6e-4 hyperparameters.lagrangian_lr=6e-4 hyperparameters.ent_target_mult=3  hyperparameters.gamma=0.9992 hyperparameters.lmbda=0.98 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl

### CheetahRun DMERL
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.lr=6e-4 hyperparameters.temperature_lr=6e-4 hyperparameters.lagrangian_lr=6e-4 hyperparameters.ent_target_mult=3  hyperparameters.gamma=0.9992 hyperparameters.lmbda=0.98 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl

### HopperStand 
#### best run so far
python -m src.jaxrl.reppo_DMERL_new env.name=HopperStand hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.lr=1e-3 hyperparameters.temperature_lr=3e-4 hyperparameters.lagrangian_lr=3e-4 hyperparameters.ent_target_mult=2.5 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl

#### not workign so well
python -m src.jaxrl.reppo_DMERL_new env.name=HopperStand hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.lr=1e-3 hyperparameters.temperature_lr=3e-4 hyperparameters.lagrangian_lr=3e-4 hyperparameters.ent_target_mult=2.5  hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl hyperparameters.use_temp_lagrangian_post_adam_ema=true hyperparameters.temp_lagrangian_ema_decay=0.999


### HopperHop ### TODO try out other optimizer, larger smoothing
python -m src.jaxrl.reppo_DMERL_new env.name=HopperHop hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.lr=1e-3 hyperparameters.temperature_lr=3e-4 hyperparameters.lagrangian_lr=3e-4 hyperparameters.ent_target_mult=2.5 hyperparameters.gamma=0.9995 hyperparameters.lmbda=0.98 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl

### WalkerRun
python -m src.jaxrl.reppo_DMERL_new env.name=WalkerRun hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.lr=1e-3 hyperparameters.temperature_lr=3e-4 hyperparameters.lagrangian_lr=3e-4 hyperparameters.ent_target_mult=2.5 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl

### WalkerRun
python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingupSparse hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.lr=1e-3 hyperparameters.temperature_lr=3e-4 hyperparameters.lagrangian_lr=3e-4 hyperparameters.ent_target_mult=2.5 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl


python -m src.jaxrl.reppo_DMERL_new \
        env.name=CheetahRun \
        wandb.project_suffix="_done_bugfix" \
        hyperparameters.num_eval=50 \
        hyperparameters.total_time_steps=50000000 \
        hyperparameters.diffusion.diff_steps=8 \
        env=mjx_dmc \
        seed=0 \
        num_trials=1 \
        experiment_overrides=mjx_dmc_large_data_dmerl_linear_schedule

### smaller batches less epochs
python -m src.jaxrl.reppo_DMERL_new env.name=WalkerRun hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_hard_envs hyperparameters.log_torso_com=true

python -m src.jaxrl.reppo_DMERL_new env.name=WalkerRun hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_WPO_hard_envs

python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingupSparse hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_hard_envs.yaml seed=3

### HopperHop
python -m src.jaxrl.reppo_DMERL_new env.name=HopperHop hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_hard_envs.yaml seed=0

python -m src.jaxrl.reppo_DMERL_new env.name=HopperHop hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_hard_envs.yaml seed=1

python -m src.jaxrl.reppo_DMERL_new env.name=HopperHop hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_hard_envs.yaml seed=2

### HumanoidWalk
python -m src.jaxrl.reppo_DMERL_new env.name=HumanoidWalk hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_hard_envs.yaml seed=0 hyperparameters.log_torso_com=true

python -m src.jaxrl.reppo_DMERL_new env.name=HopperHop hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_hard_envs.yaml seed=1

python -m src.jaxrl.reppo_DMERL_new env.name=HopperHop hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_hard_envs.yaml seed=2

### AcrobotSwingup
python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingup hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_hard_envs.yaml seed=0


### AcrobotSwingup DMERL
python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingup hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.lr=3e-4 hyperparameters.temperature_lr=3e-4 hyperparameters.lagrangian_lr=3e-4 hyperparameters.ent_target_mult=3  hyperparameters.gamma=0.9992 hyperparameters.lmbda=0.98 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl

### AcrobotSwingupSparse
python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingupSparse hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.lr=6e-4 hyperparameters.temperature_lr=6e-4 hyperparameters.lagrangian_lr=6e-4 hyperparameters.ent_target_mult=2.5  hyperparameters.gamma=0.9992 hyperparameters.lmbda=0.98 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl

### HumanoidRun DMERL
python -m src.jaxrl.reppo_DMERL_new env.name=HumanoidRun hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.lr=1e-3 hyperparameters.vmin=-20 hyperparameters.vmax=150 hyperparameters.temperature_lr=1e-3  hyperparameters.lagrangian_lr=1e-3  hyperparameters.ent_target_mult=3 hyperparameters.num_bins=171 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98  hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_humanoid experiment_overrides=mjx_humanoid_large_data_DMERL

### HumanoidWalk DMERL
python -m src.jaxrl.reppo_DMERL_new env.name=HumanoidWalk hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.lr=6e-4 hyperparameters.vmin=-20 hyperparameters.vmax=150 hyperparameters.temperature_lr=6e-4  hyperparameters.lagrangian_lr=6e-4  hyperparameters.ent_target_mult=2.5 hyperparameters.num_bins=171 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98  hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_humanoid experiment_overrides=mjx_dmc_large_data_dmerl


### run due to empty gpu
CUDA_VISIBLE_DEVICES=2 python -m src.jaxrl.reppo_DMERL_new env.name=CartpoleSwingup wandb.project_suffix="_FinalRuns_DMERL_Reppo" \hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.lr=6e-4 hyperparameters.temperature_lr=6e-4 hyperparameters.lagrangian_lr=6e-4 hyperparameters.ent_target_mult=3 hyperparameters.gamma=0.9992 hyperparameters.lmbda=0.98 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_dmc num_trials=3 experiment_overrides=mjx_dmc_large_data_dmerl 


python -m src.jaxrl.reppo_DMERL_new env.name=HumanoidWalk hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_hard_envs.yaml seed=0 hyperparameters.log_torso_com=true

#### rerun these runs and seeds
finger spin seed 3 and 7?
walkerrun 2 seeds
walkerwalk 3 seeds
