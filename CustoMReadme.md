python -m src.jaxrl.DiffReppo env=hopper cfg.normalize_env=true cfg.num_envs=1 cfg.num_steps=1 cfg.num_diffusion_steps=10

salloc -A EU-25-100 -p qgpu_exp --exclude=acn13
salloc -A EU-25-100 -p qgpu_free
salloc -A EU-25-100 -p qgpu --time=04:00:00
python config.py --RL_algo DiffPPO

python src/jaxrl/reppo.py env=humanoid_brax env.name=humanoid
conda activate REPPO
python src/jaxrl/DiffReppo.py env=humanoid_brax env.name=humanoid

python -m src.jaxrl.reppo_dime env=humanoid_brax env.name=humanoid

python -m src.jaxrl.reppo env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.action_clip_value=1. hyperparameters.env_action_clip_value=0.999

###break it
python -m src.jaxrl.reppo env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.action_clip_value=1. hyperparameters.env_action_clip_value=0.999


###PPO
python -m src.jaxrl.reppo_PPO env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000
python -m src.jaxrl.reppo_DiffPPO env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.entropy_coef=0.002

python -m src.jaxrl.ppo_mjx env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000

cd /home/it4i-sanokows/code/DIMEReppo
###dime
python -m src.jaxrl.reppo_dime env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.action_clip_value=1. hyperparameters.env_action_clip_value=1. hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=full

### humanoid
### reppo WPO
JAX_DEBUG_NANS=1 python -m src.jaxrl.reppo env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.action_clip_value=0.999 hyperparameters.env_action_clip_value=0.999 hyperparameters.lr=1e-4  hyperparameters.train_mode=WPO hyperparameters.actor_kl_clip_mode=clipped hyperparameters.ent_target_mult=0.3

JAX_DEBUG_NANS=1 python -m src.jaxrl.reppo_dime env.name=HumanoidWalk hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.lr=4e-4 hyperparameters.action_clip_value=0.999 hyperparameters.env_action_clip_value=0.999 hyperparameters.diffusion.learn_friction=false hyperparameters.aux_loss_mult=1. hyperparameters.kl_bound=0.1

JAX_DEBUG_NANS=1 python -m src.jaxrl.reppo_dime env.name=HumanoidWalk hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=full hyperparameters.lr=4e-4 hyperparameters.action_clip_value=0.999 hyperparameters.env_action_clip_value=0.999 hyperparameters.diffusion.learn_friction=false hyperparameters.aux_loss_mult=1. hyperparameters.kl_bound=0.1

### DMERL ### this works
python -m src.jaxrl.reppo_DMERL_new env.name=HumanoidWalk hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=clipped hyperparameters.lr=4e-4 hyperparameters.temperature_lr=1e-5 hyperparameters.lagrangian_lr=1e-5 hyperparameters.action_clip_value=0.999 hyperparameters.env_action_clip_value=0.999 hyperparameters.diffusion.learn_friction=false hyperparameters.aux_loss_mult=1. hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.9995 hyperparameters.lmbda=0.97 hyperparameters.weight_decay=1e-6 hyperparameters.vmin=-100 hyperparameters.vmax=250 hyperparameters.num_bins=251

python -m src.jaxrl.reppo_DMERL_new env.name=HumanoidWalk hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.lr=4e-4 hyperparameters.temperature_lr=6e-5 hyperparameters.lagrangian_lr=6e-5 hyperparameters.action_clip_value=0.999 hyperparameters.env_action_clip_value=0.999 hyperparameters.diffusion.learn_friction=false hyperparameters.aux_loss_mult=1. hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.9995 hyperparameters.lmbda=0.97 hyperparameters.weight_decay=1e-5 hyperparameters.vmin=-30 hyperparameters.vmax=200 hyperparameters.num_bins=201

python -m src.jaxrl.reppo_DMERL_new env.name=HumanoidWalk hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=clipped hyperparameters.lr=1e-3 hyperparameters.temperature_lr=3e-5 hyperparameters.lagrangian_lr=3e-5 hyperparameters.action_clip_value=0.999 hyperparameters.env_action_clip_value=0.999 hyperparameters.diffusion.learn_friction=false hyperparameters.aux_loss_mult=1. hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.9995 hyperparameters.lmbda=0.97 hyperparameters.weight_decay=1e-6 hyperparameters.vmin=-100 hyperparameters.vmax=250 hyperparameters.num_bins=251

# test
JAX_DEBUG_NANS=1 python -m src.jaxrl.reppo_DMERL_new env.name=HumanoidWalk hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=clipped hyperparameters.lr=1e-3 hyperparameters.temperature_lr=3e-5 hyperparameters.lagrangian_lr=3e-5 hyperparameters.action_clip_value=0.999 hyperparameters.env_action_clip_value=0.999 hyperparameters.diffusion.learn_friction=false hyperparameters.aux_loss_mult=1. hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.9995 hyperparameters.lmbda=0.97 hyperparameters.weight_decay=1e-3 hyperparameters.vmin=-100 hyperparameters.vmax=250 hyperparameters.num_bins=251

# test
python -m src.jaxrl.reppo_DMERL_new env.name=HumanoidWalk hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.lr=1e-3 hyperparameters.temperature_lr=3e-5 hyperparameters.lagrangian_lr=3e-5 hyperparameters.action_clip_value=0.999 hyperparameters.env_action_clip_value=0.999 hyperparameters.diffusion.learn_friction=false hyperparameters.aux_loss_mult=1. hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.9995 hyperparameters.lmbda=0.97 hyperparameters.weight_decay=1e-4 hyperparameters.vmin=-100 hyperparameters.vmax=250 hyperparameters.num_bins=251


python -m src.jaxrl.reppo_DMERL_new env.name=HumanoidWalk hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=4e-4 hyperparameters.vmin=-100 hyperparameters.vmax=250 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=3e-5 hyperparameters.lagrangian_lr=3e-5 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=251 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.9995 hyperparameters.lmbda=0.97 hyperparameters.weight_decay=1e-6 hyperparameters.lr_decay_factor=0.1 hyperparameters.num_steps=128 hyperparameters.hl_gauss=true hyperparameters.use_temp_lagrangian_mlp=false

python -m src.jaxrl.reppo env.name=HumanoidWalk hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.action_clip_value=0.999 hyperparameters.env_action_clip_value=0.999 hyperparameters.lr=5e-5

### Cheetah
#best run no critic weight decay
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-40 hyperparameters.vmax=120 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=full hyperparameters.temperature_lr=1e-5 hyperparameters.lagrangian_lr=1e-5 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=151 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.98 hyperparameters.lmbda=0.93 hyperparameters.weight_decay=1e-6

### laarger minibatch size
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-40 hyperparameters.vmax=200 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=6e-4 hyperparameters.lagrangian_lr=6e-4 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=151 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=0 hyperparameters.num_mini_batches=16

python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-40 hyperparameters.vmax=200 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=6e-4 hyperparameters.lagrangian_lr=6e-4 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=151 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.9995 hyperparameters.lmbda=0.96 hyperparameters.weight_decay=0 hyperparameters.num_mini_batches=16


### best DMLER run so far
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=4e-4 hyperparameters.vmin=-100 hyperparameters.vmax=250 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=3e-5 hyperparameters.lagrangian_lr=3e-5 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=251 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.9995 hyperparameters.lmbda=0.97 hyperparameters.weight_decay=1e-6 hyperparameters.lr_decay_factor=0.1 hyperparameters.num_steps=128 hyperparameters.hl_gauss=true hyperparameters.use_temp_lagrangian_mlp=false


###  works as well
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=5e-4 hyperparameters.vmin=-40 hyperparameters.vmax=120 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=full hyperparameters.temperature_lr=1e-5 hyperparameters.lagrangian_lr=1e-5 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=151 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.97 hyperparameters.lmbda=0.90 hyperparameters.weight_decay=0. hyperparameters.lr_decay_factor=1.

### good run without weight decay
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-40 hyperparameters.vmax=150 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=full hyperparameters.temperature_lr=1e-5 hyperparameters.lagrangian_lr=1e-5 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=151 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.97 hyperparameters.weight_decay=1e-6 hyperparameters.lr_decay_factor=0.1 hyperparameters.num_steps=128

python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-100 hyperparameters.vmax=250 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=full hyperparameters.temperature_lr=5e-6 hyperparameters.lagrangian_lr=5e-6 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=251 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.9995 hyperparameters.lmbda=0.98 hyperparameters.weight_decay=1e-6 hyperparameters.lr_decay_factor=0.1 hyperparameters.num_steps=128

python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-100 hyperparameters.vmax=250 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=full hyperparameters.temperature_lr=1e-5 hyperparameters.lagrangian_lr=1e-5 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=251 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.9998 hyperparameters.lmbda=0.97 hyperparameters.weight_decay=1e-6 hyperparameters.lr_decay_factor=0.1 hyperparameters.num_steps=128

### fkl bound works well
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-100 hyperparameters.vmax=250 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=5e-5 hyperparameters.lagrangian_lr=5e-5 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=251 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.9995 hyperparameters.lmbda=0.97 hyperparameters.weight_decay=1e-6 hyperparameters.lr_decay_factor=0.1 hyperparameters.num_steps=128 hyperparameters.hl_gauss=true hyperparameters.use_temp_lagrangian_mlp=false

### rkl bound clipped
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=4e-4 hyperparameters.vmin=-100 hyperparameters.vmax=250 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=3e-5 hyperparameters.lagrangian_lr=3e-5 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=251 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.9995 hyperparameters.lmbda=0.97 hyperparameters.weight_decay=1e-6 hyperparameters.lr_decay_factor=0.1 hyperparameters.num_steps=128 hyperparameters.hl_gauss=true hyperparameters.use_temp_lagrangian_mlp=false

### fkl 
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-100 hyperparameters.vmax=250 hyperparameters.reverse_kl=false hyperparameters.actor_kl_clip_mode=clipped hyperparameters.temperature_lr=5e-5 hyperparameters.lagrangian_lr=5e-5 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=251 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.9995 hyperparameters.lmbda=0.97 hyperparameters.weight_decay=1e-6 hyperparameters.lr_decay_factor=0.1 hyperparameters.num_steps=128 hyperparameters.hl_gauss=true hyperparameters.use_temp_lagrangian_mlp=false

### higher kl bound
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-100 hyperparameters.vmax=250 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=full hyperparameters.temperature_lr=2e-5 hyperparameters.lagrangian_lr=2e-5 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=251 hyperparameters.kl_bound=0.12 hyperparameters.gamma=0.9995 hyperparameters.lmbda=0.97 hyperparameters.weight_decay=1e-6 hyperparameters.lr_decay_factor=0.1 hyperparameters.num_steps=128 hyperparameters.hl_gauss=false hyperparameters.use_temp_lagrangian_mlp=false

python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-40 hyperparameters.vmax=150 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=full hyperparameters.temperature_lr=2e-5 hyperparameters.lagrangian_lr=2e-5 hyperparameters.ent_target_mult=6 hyperparameters.num_bins=151 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.9998 hyperparameters.lmbda=0.97 hyperparameters.weight_decay=1e-6 hyperparameters.lr_decay_factor=0.1 hyperparameters.num_steps=128 hyperparameters.num_epochs=6

python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-40 hyperparameters.vmax=150 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=full hyperparameters.temperature_lr=5e-5 hyperparameters.lagrangian_lr=5e-5 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=151 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.9998 hyperparameters.lmbda=0.97 hyperparameters.weight_decay=1e-6 hyperparameters.lr_decay_factor=0.1 hyperparameters.num_steps=128

### TODO check if diffusion is initialized so that it maps to prior


Next steps
### implement learned prior and learned std
### implement WPO and DiffPPO
### log the entropy and not lower bound estimate
### plot point of mass
### todo sweep over gamma and lambda
### todo test lr annealing

### TODO log time of the diff env steps
### TODO find out how large batch size should be!
### test if full batch yields same tmeperature updates