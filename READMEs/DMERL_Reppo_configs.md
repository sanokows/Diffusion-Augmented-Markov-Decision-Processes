


### CheetahRun DMERL with environment-time discounting
exec python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingup hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_env_time seed=0

exec python -m src.jaxrl.reppo_DMERL_new env.name=FingerSpin hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule seed=0

exec python -m src.jaxrl.reppo_DMERL_new env.name=FingerSpin hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule seed=1

exec python -m src.jaxrl.reppo_DMERL_new env.name=HopperStand hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_temp seed=2 hyperparameters.ent_target_mult=4

exec python -m src.jaxrl.reppo_DMERL_new env.name=FishSwim hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_temp seed=2 hyperparameters.ent_target_mult=4

exec python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingupSparse hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_temp seed=2 hyperparameters.ent_target_mult=4 hyperparameters.aux_loss_mult=1.


exec python -m src.jaxrl.reppo_DMERL_new env.name=WalkerRun hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=4 hyperparameters.importance_sample_diffusion_steps=true

exec python -m src.jaxrl.reppo_DMERL_new env.name=WalkerRun hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=4 hyperparameters.importance_sample_diffusion_steps=true hyperparameters.diffusion_step_sampling_mode=song

exec python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingupSparse hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=4 hyperparameters.use_final_step_reward_target=true hyperparameters.aux_loss_mult=0.75

exec python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingupSparse hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=1 hyperparameters.ent_target_mult=4 hyperparameters.use_final_step_reward_target=true hyperparameters.aux_loss_mult=0.75

exec python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingupSparse hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=4 hyperparameters.aux_loss_mult=1. hyperparameters.use_final_step_reward_target=true

exec python -m src.jaxrl.reppo_DMERL_new env.name=HumanoidRun hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_humanoid experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_humanoid seed=0 hyperparameters.ent_target_mult=1 

exec python -m src.jaxrl.reppo_DMERL_new env.name=WalkerRun hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=4 hyperparameters.diffusion.score_model.langevin_param=true


exec python -m src.jaxrl.reppo_DMERL_new env.name=HopperHop hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=4

exec python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingupSparse hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=3. hyperparameters.aux_loss_mult=0.25


exec python -m src.jaxrl.reppo_DMERL_new env.name=HopperHop hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=3. hyperparameters.aux_loss_mult=0.25


exec python -m src.jaxrl.reppo_DMERL_new env.name=HopperHop hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=3 aux_loss_mult=0.25

exec python -m src.jaxrl.reppo_DMERL_new env.name=TurningDoubleWellEnv hyperparameters.num_eval=50 hyperparameters.total_time_steps=10000000 hyperparameters.diffusion.diff_steps=8 env=mjx_double_well experiment_overrides=dmerl/mjx_double_well_DMERL


exec python -m src.jaxrl.reppo_DMERL_new env.name=FishSwim hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_kl_bound seed=0 hyperparameters.kl_bound=0.04

exec python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingupSparse hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=dmerl/mjx_dmc_large_data_dmerl_linear_schedule_kl_bound seed=0 hyperparameters.kl_bound=0.04

exec python -m src.jaxrl.reppo_DMERL_new env.name=G1JoystickFlatTerrain wandb.project_suffix=_FR_more_steps_rew_norm_final hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.num_mini_batches=2 hyperparameters.lr=1e-3 hyperparameters.temperature_lr=3e-4 hyperparameters.lagrangian_lr=3e-4 hyperparameters.gamma=0.9938 hyperparameters.lmbda=0.98 hyperparameters.vmin=-25 hyperparameters.vmax=20 hyperparameters.num_bins=301 hyperparameters.aux_loss_mult=0.05 hyperparameters.ent_target_mult=6 hyperparameters.num_collection_step_factor=0.5 hyperparameters.normalize_reward=true hyperparameters.ent_start=0.01 env=mjx_humanoid_dime num_trials=1 seed=2 experiment_overrides=dmerl/mjx_humanoid_large_data_DMERL
