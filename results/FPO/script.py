import wandb
import pandas as pd
import os

# --- CONFIGURATION ---
ENTITY = "patilkaustubh35-mit-world-peace-university"
PROJECT = "fpo_new2_baselines"
METRIC_NAME = "eval/reward_mean"
OUTPUT_CSV = "fpo_full_results.csv"

def fetch_data():
    print(f"Connecting to WandB API for {ENTITY}/{PROJECT}...")
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    all_data = []

    print(f"Found {len(runs)} runs. Downloading data...")
    for i, run in enumerate(runs):
        # 1. Extract Configs
        env = run.config.get("env_name", "Unknown")
        seed = run.config.get("seed", "Unknown")

        # 2. Extract Runtime (WandB tracks this automatically in seconds)
        runtime_seconds = run.summary.get("_runtime", 0)
        runtime_hours = runtime_seconds / 3600

        # 3. Fetch the learning curve history for this run
        # We sample it to avoid massive file sizes (e.g., 500 points per run)
        try:
            history = run.history(keys=["_step", METRIC_NAME], samples=500)

            for _, row in history.iterrows():
                all_data.append({
                    "Environment": env,
                    "Seed": seed,
                    "Step": row.get("_step", 0),
                    "Reward": row.get(METRIC_NAME, None),
                    "Runtime_Hours": runtime_hours
                })
        except Exception as e:
            print(f"Skipping run {run.name} due to error: {e}")

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(runs)} runs...")

    # 4. Save to CSV
    df = pd.DataFrame(all_data)
    # Drop rows where reward might be NaN
    df = df.dropna(subset=["Reward"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSuccess! Saved {len(df)} data points to {OUTPUT_CSV}")

if __name__ == "__main__":
    fetch_data()