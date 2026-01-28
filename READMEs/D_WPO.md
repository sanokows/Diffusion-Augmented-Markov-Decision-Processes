### PendulumSwingUp
python -m src.jaxrl.reppo env.name=PendulumSwingup env=mjx_dmc experiment_overrides=mjx_dmc_large_data_WPO hyperparameters.train_mode=WPO hyperparameters.lr=6e-4 wandb.project_suffix=_WPO_test hyperparameters.ent_target_mult=0.5

### CheetahRun
python -m src.jaxrl.reppo env.name=CheetahRun env=mjx_dmc experiment_overrides=mjx_dmc_large_data_WPO hyperparameters.train_mode=WPO hyperparameters.lr=6e-4 wandb.project_suffix=_WPO_test hyperparameters.ent_target_mult=0.5

### FingerTurnHard
python -m src.jaxrl.reppo env.name=CheetahFingerTurnHardRun env=mjx_dmc experiment_overrides=mjx_dmc_large_data_WPO hyperparameters.train_mode=WPO hyperparameters.lr=6e-4 wandb.project_suffix=_WPO_test hyperparameters.ent_target_mult=0.5

### FingerTurnEasy
python -m src.jaxrl.reppo env.name=FingerTurnEasy env=mjx_dmc experiment_overrides=mjx_dmc_large_data_WPO hyperparameters.train_mode=WPO hyperparameters.lr=6e-4 wandb.project_suffix=_WPO_test hyperparameters.ent_target_mult=0.5

### FingerSpin
python -m src.jaxrl.reppo env.name=FingerSpin env=mjx_dmc experiment_overrides=mjx_dmc_large_data_WPO hyperparameters.train_mode=WPO hyperparameters.lr=6e-4 wandb.project_suffix=_WPO_test hyperparameters.ent_target_mult=0.5

### AcrobotSwingupSparse
python -m src.jaxrl.reppo env.name=AcrobotSwingupSparse env=mjx_dmc experiment_overrides=mjx_dmc_large_data_WPO hyperparameters.train_mode=WPO hyperparameters.lr=6e-4 wandb.project_suffix=_WPO_test hyperparameters.ent_target_mult=0.5

### vary aux_loss
python -m src.jaxrl.reppo env.name=AcrobotSwingupSparse env=mjx_dmc experiment_overrides=mjx_dmc_large_data_WPO hyperparameters.train_mode=WPO hyperparameters.lr=6e-4 wandb.project_suffix=_WPO_test hyperparameters.ent_target_mult=0.5 hyperparameters.aux_loss_mult=0.01

python -m src.jaxrl.reppo env.name=AcrobotSwingupSparse env=mjx_dmc experiment_overrides=mjx_dmc_large_data_WPO hyperparameters.train_mode=WPO hyperparameters.lr=6e-4 wandb.project_suffix=_WPO_test hyperparameters.ent_target_mult=0.5 hyperparameters.aux_loss_mult=0.0

### AcrobotSwingup
python -m src.jaxrl.reppo env.name=AcrobotSwingup env=mjx_dmc experiment_overrides=mjx_dmc_large_data_WPO hyperparameters.train_mode=WPO hyperparameters.lr=6e-4 wandb.project_suffix=_WPO_test hyperparameters.ent_target_mult=0.5

python -m src.jaxrl.reppo env.name=AcrobotSwingup env=mjx_dmc experiment_overrides=mjx_dmc_large_data_WPO hyperparameters.train_mode=WPO hyperparameters.lr=6e-4 wandb.project_suffix=_WPO_test hyperparameters.disable_temperature=true

# python -m src.jaxrl.reppo env.name=AcrobotSwingup wandb.project_suffix=_FR_WPO hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 env=mjx_dmc seed=0 num_trials=5 experiment_overrides=mjx_dmc_large_data_WPO

# python -m src.jaxrl.reppo env.name=AcrobotSwingup wandb.project_suffix=_FR_ME_WPO hyperparameters.num_eval=50 hyperparameters.total_time_steps=50000000 env=mjx_dmc seed=0 num_trials=5 experiment_overrides=mjx_dmc_large_data_ME_WPO

Easy Tasks:
AcrobotSwingup
BallInCup
CartpoleBalance
CartoleSwingup
CheetahRun
FingerSpin
FingerTurnEasy
FingerTurnHard

Hard Tasks:
AcrobotSwingupSparse
PendulumSwingup
CartpoleSwingupSparse
CartpoleBalanceSparse
FingerTurnHard