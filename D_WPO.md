### PendulumSwingUp reppo
python -m src.jaxrl.reppo env.name=PendulumSwingup env=mjx_dmc experiment_overrides=mjx_dmc_large_data_WPO hyperparameters.train_mode=WPO hyperparameters.lr=6e-4 wandb.project_suffix=test hyperparameters.ent_target_mult=0.5

### PendulumSwingUp CheetahRun
python -m src.jaxrl.reppo env.name=CheetahRun env=mjx_dmc experiment_overrides=mjx_dmc_large_data_WPO hyperparameters.train_mode=WPO hyperparameters.lr=6e-4 wandb.project_suffix=test hyperparameters.ent_target_mult=0.5

### AcrobotSwingup
python -m src.jaxrl.reppo env.name=AcrobotSwingup env=mjx_dmc experiment_overrides=mjx_dmc_large_data_WPO hyperparameters.train_mode=WPO hyperparameters.lr=6e-4 wandb.project_suffix=test hyperparameters.ent_target_mult=0.5

python -m src.jaxrl.reppo env.name=AcrobotSwingup env=mjx_dmc experiment_overrides=mjx_dmc_large_data_WPO hyperparameters.train_mode=WPO hyperparameters.lr=1e-3 wandb.project_suffix=test hyperparameters.ent_target_mult=0.5

python -m src.jaxrl.reppo env.name=AcrobotSwingup env=mjx_dmc experiment_overrides=mjx_dmc_large_data_WPO hyperparameters.train_mode=WPO hyperparameters.lr=6e-4 wandb.project_suffix=test hyperparameters.ent_target_mult=0.25 hyperparameters.kl_bound=0.2

python -m src.jaxrl.reppo env.name=AcrobotSwingup env=mjx_dmc experiment_overrides=mjx_dmc_large_data_WPO hyperparameters.train_mode=WPO hyperparameters.lr=1e-3 wandb.project_suffix=test hyperparameters.ent_target_mult=0.5 hyperparameters.kl_bound=0.2
