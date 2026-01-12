### PendulumSwingUp reppo
python -m src.jaxrl.reppo env.name=PendulumSwingup env=mjx_dmc experiment_overrides=mjx_dmc_large_data
### AcrobotSwingup
python -m src.jaxrl.reppo env.name=AcrobotSwingup env=mjx_dmc experiment_overrides=mjx_dmc_large_data

### FingerTurnEasy
python -m src.jaxrl.reppo env.name=FingerTurnEasy env=mjx_dmc experiment_overrides=mjx_dmc_large_data

### HumanoidRun vmin and vmax are adjusten in overrides
python -m src.jaxrl.reppo env.name=HumanoidRun env=mjx_dmc experiment_overrides=mjx_humanoid_large_data hyperparameters.num_eval=50


