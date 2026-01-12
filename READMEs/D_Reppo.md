### PendulumSwingUp reppo
python -m src.jaxrl.reppo env.name=PendulumSwingup env=mjx_dmc experiment_overrides=mjx_dmc_large_data
### AcrobotSwingup
python -m src.jaxrl.reppo env.name=AcrobotSwingup env=mjx_dmc experiment_overrides=mjx_dmc_large_data

### FingerTurnEasy
python -m src.jaxrl.reppo env.name=FingerTurnEasy env=mjx_dmc experiment_overrides=mjx_dmc_large_data

### AcrobotSwingupSparse
python -m src.jaxrl.reppo env.name=AcrobotSwingupSparse env=mjx_dmc experiment_overrides=mjx_dmc_large_data hyperparameters.lr=3e-4 wandb.project_suffix=_WPO_test hyperparameters.ent_target_mult=0.5

### HumanoidRun 
python -m src.jaxrl.reppo env.name=HumanoidRun env=mjx_humanoid experiment_overrides=mjx_humanoid_large_data hyperparameters.num_eval=50




