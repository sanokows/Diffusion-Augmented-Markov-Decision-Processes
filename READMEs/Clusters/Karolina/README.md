# Some docu of the Karolina  CLuster
```
salloc -A EU-25-100 -p qgpu_exp
salloc -A EU-25-100 -p qgpu_free
salloc -A EU-25-100 -p qgpu --time=02:00:00
python config.py --RL_algo DiffPPO
```

https://docs.it4i.cz/en/docs/introduction


view resources:
```
it4ifree
```

git problems:
```
pushing to repo:
GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes' git push
```


this stuff has to be activated:
```
ml Python/3.9.6-GCCcore-11.2.0
ml Anaconda3/2024.02-1
ml CUDA/12.4.0
ml GCC/11.2.0
ml Mesa/21.1.7-GCCcore-11.2.0
```

```
sbatch ./READMEs/Clusters/Karolina/template.sh
sbatch ./READMEs/Clusters/Karolina/run_DMEPPO_control_1.sh
sbatch ./READMEs/Clusters/Karolina/run_DMEPPO_control_2.sh
sbatch ./READMEs/Clusters/Karolina/run_DMEPPO_low_dim.sh

sbatch ./READMEs/Clusters/Karolina/run_DMEREPPO_control_1.sh
sbatch ./READMEs/Clusters/Karolina/run_DMEREPPO_control_2.sh
sbatch ./READMEs/Clusters/Karolina/run_DMEREPPO_low_dim.sh

sbatch ./READMEs/Clusters/Karolina/run_DMERPPO_more_steps_no_learn.sh
sbatch ./READMEs/Clusters/Karolina/run_DMERPPO_more_steps.sh

sbatch ./READMEs/Clusters/Karolina/DPPO/control_1.sh
sbatch ./READMEs/Clusters/Karolina/DPPO/control_2.sh
sbatch ./READMEs/Clusters/Karolina/DPPO/low_dim.sh
sbatch ./READMEs/Clusters/Karolina/WPO/aux_loss.sh
sbatch ./READMEs/Clusters/Karolina/DME_REPPO/rerun.sh

sbatch ./READMEs/Clusters/Karolina/WPO/control_1.sh
sbatch ./READMEs/Clusters/Karolina/WPO/control_2.sh
sbatch ./READMEs/Clusters/Karolina/WPO/low_dim.sh

sbatch ./READMEs/Clusters/Karolina/WPO/control_1_Finger.sh
sbatch ./READMEs/Clusters/Karolina/WPO/control_1_Walk.sh
sbatch ./READMEs/Clusters/Karolina/DME_REPPO/control_1.sh
sbatch ./READMEs/Clusters/Karolina/DME_REPPO/control_2.sh
sbatch ./READMEs/Clusters/Karolina/DME_REPPO/low_dim.sh
sbatch ./READMEs/Clusters/Karolina/DME_REPPO/env_time_fish_hopper.sh

sbatch ./READMEs/Clusters/Karolina/DME_REPPO/run_sweep_4_steps_tests.sh

```
