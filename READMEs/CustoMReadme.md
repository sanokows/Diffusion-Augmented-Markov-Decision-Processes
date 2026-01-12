# env names
PendulumSwingup
AcrobotSwingup
AcrobotSwingupSparse
BallInCup
CartpoleBalance
CartpoleBalanceSparse
CartoleSwingup
CartpoleSwingupSparse
CheetahRun
FingerSpin
FingerTurnEasy
FingerTurnHard


python -m src.jaxrl.DiffReppo env=hopper cfg.normalize_env=true cfg.num_envs=1 cfg.num_steps=1 cfg.num_diffusion_steps=10

salloc -A EU-25-100 -p qgpu_exp --exclude=acn13
salloc -A EU-25-100 -p qgpu_free
salloc -A EU-25-100 -p qgpu --time=02:00:00
python config.py --RL_algo DiffPPO

### TODO check if diffusion is initialized so that it maps to prior

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

Next steps
### implement learned prior and learned std
### plot point of mass

### TODO log time of the diff env steps
### TODO vecotrize dt for WPO
### TODO find out how large batch size should be!

### todo make friction learnable by neural network
### todo make prior learnable in DMERL


### TODO add distributional value function in DiffPPO
### TODO implement Q function guidance


python -m src.jaxrl.reppo env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.action_clip_value=1. hyperparameters.env_action_clip_value=0.999

###break it
python -m src.jaxrl.reppo env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.action_clip_value=1. hyperparameters.env_action_clip_value=0.999


###PPO
python -m src.jaxrl.reppo_PPO env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000
python -m src.jaxrl.reppo_DiffPPO env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.num_mini_batches=32 hyperparameters.lr=1e-4 hyperparameters.entropy_coef=0.002 hyperparameters.use_kl_regularization=false hyperparameters.num_envs=1024 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true

python -m src.jaxrl.ppo_mjx env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000

cd /home/it4i-sanokows/code/DIMEReppo

### dime
python -m src.jaxrl.reppo_dime env.name=CheetahRun hyperparameters.lr=3e-4 hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.ent_start=0.01 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.num_bins=301  hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true


### humanoid
### reppo WPO
JAX_DEBUG_NANS=1 python -m src.jaxrl.reppo_dime env.name=HumanoidWalk hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.lr=4e-4 hyperparameters.action_clip_value=0.999 hyperparameters.env_action_clip_value=0.999 hyperparameters.diffusion.learn_friction=false hyperparameters.aux_loss_mult=1. hyperparameters.kl_bound=0.1

JAX_DEBUG_NANS=1 python -m src.jaxrl.reppo_dime env.name=HumanoidWalk hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=full hyperparameters.lr=4e-4 hyperparameters.action_clip_value=0.999 hyperparameters.env_action_clip_value=0.999 hyperparameters.diffusion.learn_friction=false hyperparameters.aux_loss_mult=1. hyperparameters.kl_bound=0.1

python -m src.jaxrl.reppo_DMERL_new env.name=HumanoidWalk hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=3e-4 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=3e-4 hyperparameters.lagrangian_lr=3e-4 hyperparameters.ent_target_mult=2.5 hyperparameters.num_bins=301 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.num_mini_batches=32 hyperparameters.diffusion.learn_friction=true


python -m src.jaxrl.reppo env.name=HumanoidWalk hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.action_clip_value=0.999 hyperparameters.env_action_clip_value=0.999 hyperparameters.lr=5e-5

### Cheetah

### learn friction with mlp  code change WPO  "works"
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=3e-4 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=3e-4 hyperparameters.lagrangian_lr=3e-4 hyperparameters.ent_target_mult=3 hyperparameters.num_bins=301 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.num_mini_batches=32 hyperparameters.hl_gauss=true hyperparameters.ent_start=0.01 hyperparameters.diffusion.learn_friction=false hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.train_mode=WPO

python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=3e-4 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=3e-4 hyperparameters.lagrangian_lr=3e-4 hyperparameters.ent_target_mult=3 hyperparameters.num_bins=301 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.num_mini_batches=32 hyperparameters.hl_gauss=true hyperparameters.ent_start=0.01 hyperparameters.diffusion.learn_friction=false hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.train_mode=WPO hyperparameters.fisher_type=kfac

# Final Configs

### PendulumSwingUp DMERL
python -m src.jaxrl.reppo_DMERL_new env.name=PendulumSwingup hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-50 hyperparameters.vmax=150 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=6e-4 hyperparameters.lagrangian_lr=6e-4 hyperparameters.ent_target_mult=2.5 hyperparameters.num_bins=201 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.use_friction_mlp=false hyperparameters.hl_gauss=true hyperparameters.ent_start=0.01 hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl hyperparameters.temp_lagrangian_adam_gamma1=0.97 hyperparameters.temp_lagrangian_adam_gamma2=0.9997

### changed num_collection_step_factor and other default params
python -m src.jaxrl.reppo_DMERL_new env.name=PendulumSwingup hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=2e-3 hyperparameters.vmin=-50 hyperparameters.vmax=150 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=2e-3 hyperparameters.lagrangian_lr=2e-3 hyperparameters.ent_target_mult=2.5 hyperparameters.num_bins=201 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.use_friction_mlp=false hyperparameters.hl_gauss=true hyperparameters.ent_start=0.01 hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl


### PendulumSwingUp reppo_dime
python -m src.jaxrl.reppo_dime env.name=PendulumSwingup hyperparameters.num_eval=100 hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.ent_start=0.01 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.num_bins=301  hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.lr=3e-4 env=mjx_dmc experiment_overrides=mjx_dmc_large_data 

python -m src.jaxrl.reppo_dime env.name=AcrobotSwingup hyperparameters.num_eval=100 hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.ent_start=0.01 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.num_bins=301  hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.lr=3e-4 env=mjx_dmc experiment_overrides=mjx_dmc_large_data 

### PendulumSwingUp reppo
python -m src.jaxrl.reppo env.name=PendulumSwingup env=mjx_dmc experiment_overrides=mjx_dmc_large_data
### AcrobotSwingup
python -m src.jaxrl.reppo env.name=AcrobotSwingup env=mjx_dmc experiment_overrides=mjx_dmc_large_data

### AcrobotSwingup DMERL
python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingup hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=2e-3 hyperparameters.vmin=-50 hyperparameters.vmax=150 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=2e-3 hyperparameters.lagrangian_lr=2e-3 hyperparameters.ent_target_mult=3 hyperparameters.num_bins=201 hyperparameters.gamma=0.9992 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.use_friction_mlp=false hyperparameters.hl_gauss=true hyperparameters.ent_start=0.01 hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl

### smaller lr
python -m src.jaxrl.reppo_DMERL_new env.name=AcrobotSwingup hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=1e-3 hyperparameters.vmin=-50 hyperparameters.vmax=150 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=1e-3 hyperparameters.lagrangian_lr=1e-3 hyperparameters.ent_target_mult=3 hyperparameters.num_bins=201 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.use_friction_mlp=false hyperparameters.hl_gauss=true hyperparameters.ent_start=0.01 hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true env=mjx_dmc experiment_overrides=mjx_dmc_large_data_dmerl

### AcrobotSwingup DiffPPO
python -m src.jaxrl.reppo_DiffPPO env.name=AcrobotSwingup hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.num_mini_batches=32 hyperparameters.lr=3e-4 hyperparameters.entropy_coef=0.01 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.temp_lagrangian_adam_gamma1=0.97 hyperparameters.temp_lagrangian_adam_gamma2=0.9997 hyperparameters.num_collection_step_factor=0.5 hyperparameters.num_envs=1024 hyperparameters.num_epochs=8

### AcrobotSwingup PPO works quite well
python -m src.jaxrl.reppo_PPO env.name=AcrobotSwingup hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.num_envs=1024 hyperparameters.entropy_coef=0.01

### AcrobotSwingup PPO works quite well
python -m src.jaxrl.reppo_PPO env.name=BallInCup hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.num_envs=1024 hyperparameters.entropy_coef=0.01

### CheetahRun DiffPPO
python -m src.jaxrl.reppo_DiffPPO env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.num_mini_batches=32 hyperparameters.lr=3e-4 hyperparameters.entropy_coef=0.001 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.temp_lagrangian_adam_gamma1=0.97 hyperparameters.temp_lagrangian_adam_gamma2=0.9997 hyperparameters.num_collection_step_factor=0.5 hyperparameters.num_envs=1024 hyperparameters.num_epochs=8 hyperparameters.update_entropy_lagrangian=true

### CheetahRun DiffPPO categoricalValue
python -m src.jaxrl.reppo_DiffPPO env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.num_mini_batches=32 hyperparameters.lr=3e-4 hyperparameters.entropy_coef=0.001 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.temp_lagrangian_adam_gamma1=0.97 hyperparameters.temp_lagrangian_adam_gamma2=0.9997 hyperparameters.num_collection_step_factor=0.5 hyperparameters.num_envs=1024 hyperparameters.num_epochs=8 hyperparameters.use_categorical_value=true  hyperparameters.vmin=-50 hyperparameters.vmax=150 hyperparameters.num_bins=201

### BallInCup DiffPPO
python -m src.jaxrl.reppo_DiffPPO env.name=BallInCup hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.num_mini_batches=32 hyperparameters.lr=1e-3 hyperparameters.entropy_coef=0.001 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.learn_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.diffusion.per_step_dt=true hyperparameters.temp_lagrangian_adam_gamma1=0.97 hyperparameters.temp_lagrangian_adam_gamma2=0.9997 hyperparameters.num_collection_step_factor=0.5 hyperparameters.num_envs=1024 hyperparameters.num_epochs=8

