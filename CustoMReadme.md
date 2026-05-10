# salloc -A EU-25-100 -p qgpu --time=1:00:30 
### TODO in Q function process states, noisy actions and actions separately. -> then change state reconst loss
### then store state embedding during rollout

### TODO rerun PPO exp with smaller temp
### TODO entropy reg should also be fixed for WPO temp
### log log Z

# 1) Clone your current env
conda deactivate
conda create -n REPPO_updated --clone REPPO -y

# 2) Activate clone
conda activate REPPO_updated

# 3) Upgrade target packages
python -m pip install --upgrade "playground==0.2.0" "brax==0.14.2"

# 4) Verify versions + dependency health
python - <<'PY'
import importlib.metadata as md
for p in ["playground", "brax", "jax", "jaxlib", "mujoco_mjx"]:
    try:
        print(f"{p}=={md.version(p)}")
    except Exception:
        print(f"{p}: not installed")
PY
python -m pip check
