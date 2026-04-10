

the wand sweeps can be run with for example:

```

sbatch ./READMEs/Clusters/Meluxina/DME-PPO/DME_PPO_1.sh
sbatch ./READMEs/Clusters/Meluxina/DME-PPO/DME_PPO_low_dim.sh
sbatch ./READMEs/Clusters/Meluxina/DME-PPO/DME_PPO_2.sh
```



# Some docu of the Meluxina CLuster
myquota


install modules:
```
srun --partition gpu -t 01:00:00 -A p201037 -q default --pty /bin/bash -l
```
-> squeue --me -> conenct to node