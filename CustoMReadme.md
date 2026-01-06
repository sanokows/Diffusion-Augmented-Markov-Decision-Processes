python -m src.jaxrl.DiffReppo env=hopper cfg.normalize_env=true cfg.num_envs=1 cfg.num_steps=1 cfg.num_diffusion_steps=10

salloc -A EU-25-100 -p qgpu_exp --exclude=acn13
salloc -A EU-25-100 -p qgpu_free
salloc -A EU-25-100 -p qgpu --time=04:00:00
python config.py --RL_algo DiffPPO


python -m src.jaxrl.reppo env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.action_clip_value=1. hyperparameters.env_action_clip_value=0.999

###break it
python -m src.jaxrl.reppo env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.action_clip_value=1. hyperparameters.env_action_clip_value=0.999


###PPO
python -m src.jaxrl.reppo_PPO env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000
python -m src.jaxrl.reppo_DiffPPO env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.num_mini_batches=32 hyperparameters.lr=5e-5 hyperparameters.update_entropy_lagrangian=true hyperparameters.entropy_coef=0.002 hyperparameters.use_kl_regularization=true hyperparameters.num_envs=1024

python -m src.jaxrl.ppo_mjx env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000

cd /home/it4i-sanokows/code/DIMEReppo

### dime
python -m src.jaxrl.reppo_dime env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.action_clip_value=1. hyperparameters.env_action_clip_value=1. hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped

### humanoid
### reppo WPO
JAX_DEBUG_NANS=1 python -m src.jaxrl.reppo_dime env.name=HumanoidWalk hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.lr=4e-4 hyperparameters.action_clip_value=0.999 hyperparameters.env_action_clip_value=0.999 hyperparameters.diffusion.learn_friction=false hyperparameters.aux_loss_mult=1. hyperparameters.kl_bound=0.1

JAX_DEBUG_NANS=1 python -m src.jaxrl.reppo_dime env.name=HumanoidWalk hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=full hyperparameters.lr=4e-4 hyperparameters.action_clip_value=0.999 hyperparameters.env_action_clip_value=0.999 hyperparameters.diffusion.learn_friction=false hyperparameters.aux_loss_mult=1. hyperparameters.kl_bound=0.1

python -m src.jaxrl.reppo_DMERL_new env.name=HumanoidWalk hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=3e-4 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=3e-4 hyperparameters.lagrangian_lr=3e-4 hyperparameters.ent_target_mult=2.5 hyperparameters.num_bins=301 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.num_mini_batches=32 hyperparameters.diffusion.learn_friction=true


python -m src.jaxrl.reppo env.name=HumanoidWalk hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.action_clip_value=0.999 hyperparameters.env_action_clip_value=0.999 hyperparameters.lr=5e-5

### Cheetah
### good run with changed entropy
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-40 hyperparameters.vmax=200 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=6e-4 hyperparameters.lagrangian_lr=6e-4 hyperparameters.ent_target_mult=3 hyperparameters.num_bins=151 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.num_mini_batches=32

### with WPO
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=6e-4 hyperparameters.lagrangian_lr=6e-4 hyperparameters.ent_target_mult=3 hyperparameters.num_bins=300 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.num_mini_batches=32 hyperparameters.train_mode="WPO"

### learn friction additionally
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=6e-4 hyperparameters.lagrangian_lr=6e-4 hyperparameters.ent_target_mult=2.5 hyperparameters.num_bins=301 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.num_mini_batches=32 hyperparameters.diffusion.learn_friction=true

### learn friction with mlp
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=6e-4 hyperparameters.lagrangian_lr=6e-4 hyperparameters.ent_target_mult=2.5 hyperparameters.num_bins=301 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.num_mini_batches=32 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.use_friction_mlp=true hyperparameters.hl_gauss=true 

### more minibatches
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=2e-4 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=2e-4 hyperparameters.lagrangian_lr=2e-4 hyperparameters.ent_target_mult=2.5 hyperparameters.num_bins=300 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.num_mini_batches=64 hyperparameters.diffusion.learn_friction=true

### more minibatches with langevin_param
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=4e-4 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=4e-4 hyperparameters.lagrangian_lr=4e-4 hyperparameters.ent_target_mult=2.5 hyperparameters.num_bins=300 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.num_mini_batches=32 hyperparameters.diffusion.learn_friction=true hyperparameters.diffusion.score_model.langevin_param=true


### PendulumSwingUp
python -m src.jaxrl.reppo_DMERL_new env.name=PendulumSwingup hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=5e-5 hyperparameters.vmin=-100 hyperparameters.vmax=200 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=5e-5 hyperparameters.lagrangian_lr=5e-5 hyperparameters.ent_target_mult=2.5 hyperparameters.num_bins=301 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.num_mini_batches=32 hyperparameters.diffusion.learn_friction=true

### reppo
python -m src.jaxrl.reppo env.name=PendulumSwingup hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.action_clip_value=0.999 hyperparameters.env_action_clip_value=0.999 hyperparameters.lr=3e-4
### TODO check if diffusion is initialized so that it maps to prior

envs:
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

Next steps
### implement learned prior and learned std
### plot point of mass

### TODO log time of the diff env steps
### TODO find out how large batch size should be!

### todo make friction learnable by neural network
### todo make prior learnable in DMERL


### TODO add distributional value function in DiffPPO
### TODO implement Q function guidance