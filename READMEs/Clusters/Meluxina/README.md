

the wand sweeps can be run with for example:

```
sbatch ./READMEs/Clusters/Meluxina/dmerl_hyper.sh
sbatch ./READMEs/Clusters/Meluxina/dmerl_runs.sh
sbatch ./READMEs/Clusters/Meluxina/dmerl_low_dim.sh
sbatch ./READMEs/Clusters/Meluxina/dmerl_control.sh
sbatch ./READMEs/Clusters/Meluxina/dmerl_WPO_control.sh
sbatch ./READMEs/Clusters/Meluxina/dmerl_WPO_low_dim.sh
sbatch ./READMEs/Clusters/Meluxina/dime_runs_2.sh
sbatch ./READMEs/Clusters/Meluxina/dmerl_reppo_all.sh
sbatch ./READMEs/Clusters/Meluxina/dmerl_reruns_1.sh
```



# Some docu of the Meluxina CLuster
myquota


install modules:
```
srun --partition gpu -t 01:00:00 -A p201037 -q default --pty /bin/bash -l
```
-> squeue --me -> conenct to node