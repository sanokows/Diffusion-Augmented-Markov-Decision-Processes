
salloc --partition=gpu --nodes=1  --time=00:00:30 --gres=gpu:4 --account d2025d09-019-users


--account d2023d12-053-users


users:
d2023d12-053-users
d2025d09-019-users

flag --account

```
sbatch ./READMEs/Clusters/Vega/run_gpu_dime_control.sh
sbatch ./READMEs/Clusters/Vega/run_gpu_dime_low_dim.sh
```