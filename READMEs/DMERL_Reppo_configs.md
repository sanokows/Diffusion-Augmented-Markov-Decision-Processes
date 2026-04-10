


exec python -m src.jaxrl.reppo_DMERL_new env.name=WalkerRun hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=4

exec python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingupSparse hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=4 hyperparameters.use_final_step_reward_target=true

exec python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingupSparse hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=4

exec python -m src.jaxrl.reppo_DMERL_new env.name=HumanoidRun hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_humanoid experiment_overrides=mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=4 +env.config.impl=jax

exec python -m src.jaxrl.reppo_DMERL_new env.name=WalkerRun hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=4 hyperparameters.diffusion.score_model.langevin_param=true


exec python -m src.jaxrl.reppo_DMERL_new env.name=HopperHop hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=4

exec python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingupSparse hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=3. hyperparameters.aux_loss_mult=0.25


exec python -m src.jaxrl.reppo_DMERL_new env.name=HopperHop hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=3. hyperparameters.aux_loss_mult=0.25


exec python -m src.jaxrl.reppo_DMERL_new env.name=HopperHop hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl_linear_schedule_no_temp seed=0 hyperparameters.ent_target_mult=3 aux_loss_mult=0.25

exec python -m src.jaxrl.reppo_DMERL_new env.name=TurningDoubleWellEnv hyperparameters.num_eval=50 hyperparameters.total_time_steps=10000000 hyperparameters.diffusion.diff_steps=8 env=mjx_double_well experiment_overrides=mjx_double_well_DMERL
