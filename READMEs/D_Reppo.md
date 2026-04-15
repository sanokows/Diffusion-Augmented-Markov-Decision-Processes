### PendulumSwingUp reppo
python -m src.jaxrl.reppo env.name=PendulumSwingup env=mjx_dmc experiment_overrides=reppo/mjx_dmc_large_data


python -m src.jaxrl.reppo env.name=PlanarPathEnv env=mjx_dmc experiment_overrides=reppo/mjx_dmc_large_data

### AcrobotSwingup
python -m src.jaxrl.reppo env.name=AcrobotSwingup env=mjx_dmc experiment_overrides=reppo/mjx_dmc_large_data

### FingerTurnEasy
python -m src.jaxrl.reppo env.name=FingerTurnEasy env=mjx_dmc experiment_overrides=reppo/mjx_dmc_large_data

### AcrobotSwingupSparse
python -m src.jaxrl.reppo env.name=AcrobotSwingupSparse env=mjx_dmc experiment_overrides=reppo/mjx_dmc_large_data hyperparameters.lr=3e-4 wandb.project_suffix=_WPO_test hyperparameters.ent_target_mult=0.5

### HopperStand
python -m src.jaxrl.reppo env.name=HopperStand env=mjx_dmc experiment_overrides=reppo/mjx_dmc_large_data

### HopperHop
python -m src.jaxrl.reppo env.name=HopperHop env=mjx_humanoid experiment_overrides=reppo/mjx_humanoid_large_data

### WalkerRun
exec python -m src.jaxrl.reppo env.name=FishSwim env=mjx_dmc experiment_overrides=reppo/mjx_dmc_large_data

### HumanoidRun 
python -m src.jaxrl.reppo env.name=HumanoidWalk env=mjx_humanoid experiment_overrides=reppo/mjx_humanoid_large_data hyperparameters.num_eval=50

python -m src.jaxrl.reppo env.name=HumanoidWalk env=mjx_humanoid experiment_overrides=reppo/mjx_humanoid_large_data hyperparameters.num_eval=50

exec  python -m src.jaxrl.reppo env.name=TurningDoubleWellEnv env=mjx_double_well experiment_overrides=reppo/mjx_double_well_large_data hyperparameters.num_eval=50


exec  python -m src.jaxrl.reppo env.name=TurningGMMEnv env=mjx_gmm experiment_overrides=reppo/mjx_double_well_large_data hyperparameters.num_eval=50



