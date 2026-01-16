### AcrobotSwingup PPO works quite well
python -m src.jaxrl.reppo_PPO env.name=AcrobotSwingup hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.num_envs=1024 hyperparameters.entropy_coef=0.01


### HopperStand 
python -m src.jaxrl.reppo_PPO env.name=HopperStand hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.num_envs=1024 hyperparameters.entropy_coef=0.00000001

### HopperStand noentropy
python -m src.jaxrl.reppo_PPO env.name=HopperStand hyperparameters.num_eval=100 hyperparameters.total_time_steps=50000000 hyperparameters.num_envs=1024 