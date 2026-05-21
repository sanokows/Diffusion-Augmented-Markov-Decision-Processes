

the wand sweeps can be run with for example:

```

sbatch ./READMEs/Clusters/Meluxina/DME-PPO/DME_PPO_1.sh
sbatch ./READMEs/Clusters/Meluxina/DME-PPO/DME_PPO_low_dim.sh
sbatch ./READMEs/Clusters/Meluxina/DME-PPO/DME_PPO_2.sh
sbatch ./READMEs/Clusters/Meluxina/DME-PPO/DME_PPO_hyperparam.sh
sbatch ./READMEs/Clusters/Meluxina/DME-REPPO_Sparse/run.sh


sbatch ./READMEs/Clusters/Meluxina/DPPO/DPPO_1.sh
sbatch ./READMEs/Clusters/Meluxina/DPPO/DPPO_2.sh
sbatch ./READMEs/Clusters/Meluxina/DPPO/DPPO_low_dim.sh

sbatch ./READMEs/Clusters/Meluxina/DA_MDP_WPO/control_1.sh
sbatch ./READMEs/Clusters/Meluxina/DA_MDP_WPO/control_2.sh
sbatch ./READMEs/Clusters/Meluxina/DA_MDP_WPO/low_dim.sh
sbatch ./READMEs/Clusters/Meluxina/DA_MDP_WPO/low_dim_test.sh

sbatch ./READMEs/Clusters/Meluxina/REPPO/all.sh
```



# Some docu of the Meluxina CLuster
myquota


install modules:
```
srun --partition gpu -t 01:00:00 -A p201037 -q default --pty /bin/bash -l
```
-> squeue --me -> conenct to node