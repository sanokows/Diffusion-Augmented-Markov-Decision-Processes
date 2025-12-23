python -m src.jaxrl.DiffReppo env=hopper cfg.normalize_env=true cfg.num_envs=1 cfg.num_steps=1 cfg.num_diffusion_steps=10

salloc -A EU-25-100 -p qgpu_exp --exclude=acn13
salloc -A EU-25-100 -p qgpu_free
salloc -A EU-25-100 -p qgpu --time=02:00:00
python config.py --RL_algo DiffPPO

python src/jaxrl/reppo.py env=humanoid_brax env.name=humanoid
conda activate REPPO
python src/jaxrl/DiffReppo.py env=humanoid_brax env.name=humanoid

python -m src.jaxrl.reppo_dime env=humanoid_brax env.name=humanoid

python -m src.jaxrl.reppo env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.action_clip_value=1. hyperparameters.env_action_clip_value=0.999

###break it
python -m src.jaxrl.reppo env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.action_clip_value=1. hyperparameters.env_action_clip_value=0.999


cd /home/it4i-sanokows/code/DIMEReppo
python -m src.jaxrl.reppo_dime env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.action_clip_value=1. hyperparameters.env_action_clip_value=1.

#best run no critic weight decay
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-40 hyperparameters.vmax=120 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=full hyperparameters.temperature_lr=1e-5 hyperparameters.lagrangian_lr=1e-5 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=151 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.98 hyperparameters.lmbda=0.93 hyperparameters.weight_decay=1e-6


### tests
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=6e-4 hyperparameters.vmin=-40 hyperparameters.vmax=120 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=full hyperparameters.temperature_lr=1e-5 hyperparameters.lagrangian_lr=1e-5 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=151 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.97 hyperparameters.lmbda=0.90 hyperparameters.weight_decay=1e-6

###  works as well
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=5e-4 hyperparameters.vmin=-40 hyperparameters.vmax=120 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=full hyperparameters.temperature_lr=1e-5 hyperparameters.lagrangian_lr=1e-5 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=151 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.97 hyperparameters.lmbda=0.90 hyperparameters.weight_decay=0. hyperparameters.lr_decay_factor=1.

### good run without weight decay
python -m src.jaxrl.reppo_DMERL_new env.name=CheetahRun hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=1 hyperparameters.lr=5e-4 hyperparameters.vmin=-40 hyperparameters.vmax=120 hyperparameters.reverse_kl=true hyperparameters.actor_kl_clip_mode=full hyperparameters.temperature_lr=1e-5 hyperparameters.lagrangian_lr=1e-5 hyperparameters.ent_target_mult=4 hyperparameters.num_bins=151 hyperparameters.kl_bound=0.1 hyperparameters.gamma=0.999 hyperparameters.lmbda=0.97 hyperparameters.weight_decay=0. hyperparameters.lr_decay_factor=1. hyperparameters.num_steps=128

### TODO check if diffusion is initialized so that it maps to prior


Next steps
### implement learned prior and learned std
### implement WPO and DiffPPO
### log the entropy and not lower bound estimate
### plot point of mass
### todo sweep over gamma and lambda
### todo test lr annealing