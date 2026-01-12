

the wand sweeps can be run with for example:

```
sbatch ./Clusters/Melux/DiffSteps/run_gpu_DiffSAC.sh
sbatch ./Clusters/Melux/Temperature/run_gpu_DiffSAC.sh
sbatch ./Clusters/Melux/Temperature/run_gpu_DiffPPO.sh
sbatch ./Clusters/Melux/Ant/run_gpu_DiffSAC.sh
```



# Some docu of the Meluxina CLuster
myquota


install modules:
```
srun --partition gpu -t 01:00:00 -A p201037 -q default --pty /bin/bash -l
```
-> squeue --me -> conenct to node