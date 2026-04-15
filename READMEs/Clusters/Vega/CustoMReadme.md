
salloc --partition=gpu --nodes=1  --time=01:00:30 --gres=gpu:1 --account d2025d09-019-users


--account d2023d12-053-users


users:
d2023d12-053-users
d2025d09-019-users

flag --account

```
sbatch ./READMEs/Clusters/Vega/run_gpu_dime_control.sh
sbatch ./READMEs/Clusters/Vega/run_gpu_dime_low_dim.sh
sbatch ./READMEs/Clusters/Vega/run_gpu_dmerl_control_1.sh
sbatch ./READMEs/Clusters/Vega/run_gpu_dmerl_control_2.sh
sbatch ./READMEs/Clusters/Vega/run_gpu_dmerl_low_dim.sh
sbatch ./READMEs/Clusters/Vega/run_gpu_WPO_hyper.sh

sbatch ./READMEs/Clusters/Vega/run_gpu_dmerl_hard_envs.sh
sbatch ./READMEs/Clusters/Vega/run_gpu_dmerl_WPO_hard_envs.sh
sbatch ./READMEs/Clusters/Vega/run_gpu_reppo_all.sh

sbatch ./READMEs/Clusters/Vega/run_gpu_dmerl_WPO_control_1.sh
sbatch ./READMEs/Clusters/Vega/run_gpu_dmerl_WPO_control_2.sh
sbatch ./READMEs/Clusters/Vega/run_gpu_dmerl_WPO_low_dim.sh
sbatch ./READMEs/Clusters/Vega/run_gpu_dmerl_WPO_hard_envs.sh
sbatch ./READMEs/Clusters/Vega/run_gpu_dmerl_humanoid.sh
sbatch ./READMEs/Clusters/Vega/run_gpu_dime_humanoid.sh

sbatch ./READMEs/Clusters/Vega/run_gpu_me_wpo_all.sh
sbatch ./READMEs/Clusters/Vega/run_gpu_wpo_all.sh

sbatch ./READMEs/Clusters/Vega/run_gpu_dmerl_more_steps.sh
sbatch ./READMEs/Clusters/Vega/run_gpu_dmerl_WPO_walker.sh

sbatch --exclude=gn51 ./READMEs/Clusters/Vega/run_gpu_dmerl_aux_loss.sh 
sbatch ./READMEs/Clusters/Vega/run_gpu_dmerl_WPO_aux_loss.sh
sbatch ./READMEs/Clusters/Vega/run_gpu_dmerl_arcsparse.sh
sbatch ./READMEs/Clusters/Vega/run_gpu_dmerl_wpo_arcsparse.sh

sbatch ./READMEs/Clusters/Vega/run_gpu_dmerl_more_steps_no_learn_diff.sh

sbatch ./READMEs/Clusters/Vega/run_gpu_dmerl_more_steps.sh
sbatch ./READMEs/Clusters/Vega/FinalStepRuns/run_gpu_dmerl_4_steps.sh

sbatch ./READMEs/Clusters/Vega/FinalStepRuns/run_gpu_dmerl_8_steps.sh
sbatch ./READMEs/Clusters/Vega/Rebuttal/run_gpu_dmerl_small_nets_vmapped.sh
sbatch ./READMEs/Clusters/Vega/Rebuttal/run_gpu_dppo_entropy_sweep.sh


sbatch ./READMEs/Clusters/Vega/DME_REPPO/run_gpu_dmerl_control_1.sh
sbatch ./READMEs/Clusters/Vega/DME_REPPO/run_gpu_dmerl_control_2.sh
sbatch ./READMEs/Clusters/Vega/DME_REPPO/run_gpu_dmerl_low_dim.sh
```