python -m src.jaxrl.DiffReppo env=hopper cfg.normalize_env=true cfg.num_envs=1 cfg.num_steps=1 cfg.num_diffusion_steps=10

salloc -A EU-25-100 -p qgpu_exp
salloc -A EU-25-100 -p qgpu_free
salloc -A EU-25-100 -p qgpu --time=02:00:00
python config.py --RL_algo DiffPPO

python src/jaxrl/reppo.py env=humanoid_brax env.name=humanoid
conda activate REPPO
python src/jaxrl/DiffReppo.py env=humanoid_brax env.name=humanoid

python -m src.jaxrl.reppo_dime env=humanoid_brax env.name=humanoid


cd /home/it4i-sanokows/code/DIMEReppo
python -m src.jaxrl.reppo_dime env.name=CheetahRun hyperparameters.num_eval=10 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4


python -m src.jaxrl.reppo_DMERL env.name=CheetahRun hyperparameters.num_eval=10 hyperparameters.total_time_steps=50000000 hyperparameters.diffusion.diff_steps=8 hyperparameters.kl_action_rep=4
