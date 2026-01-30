import itertools
import subprocess
import yaml
import copy
from pathlib import Path

# -------------------------------------------------------------------
# All available MovingWindow tasks
# -------------------------------------------------------------------
ALL_TASKS = [
    dict(
        name="MovingWindowSum",
        sep=17,
        n_train=128,
        n_test=32,
        min_num=1,
        max_num=16,
        k=2,
        p=17,
    ),
    dict(
        name="MovingWindowDifference",
        sep=18,
        n_train=128,
        n_test=32,
        min_num=1,
        max_num=16,
        k=2,
        p=17,
    ),
    dict(
        name="MovingWindowProduct",
        sep=19,
        n_train=128,
        n_test=32,
        min_num=1,
        max_num=16,
        k=2,
        p=17,
    ),
    # dict(
    #     name="MovingWindowQuotient",
    #     sep=20,
    #     n_train=128,
    #     n_test=32,
    #     min_num=1,
    #     max_num=16,
    #     k=2,
    #     p=17,
    # ),
]

# -------------------------------------------------------------------
# Base config template (shared across all runs)
# -------------------------------------------------------------------
BASE_CONFIG = {
    "model": {
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 256,
        "linear": True,
    },
    "data": {
        "tasks": [],   # filled dynamically
        "min_num": 1,
        "max_num": 16,
        "k": 2,
        "p": 17,
        "num_tokens": 16,
        "fixed_len": True,
        "mix": "random",
    },
    "train": {
        "lr": 1e-4,
        "grad_clip": -1,
        "num_steps": 2,
        "norm_type": "none_rank",
        "wandb": True,
        "wandb_project": "loss_plateau_tf",
        "save_ckpt": False,
        "ckpt_freq": 20,
        "seed": 67,
        "mask_input": True,
        "freeze_embedding": True,
    },
}

# -------------------------------------------------------------------
# Generate all non-empty task subsets
# -------------------------------------------------------------------
def all_task_subsets(tasks):
    subsets = []
    for r in range(1, len(tasks) + 1):
        subsets.extend(itertools.combinations(tasks, r))
    return subsets

# -------------------------------------------------------------------
# Main sweep logic
# -------------------------------------------------------------------
def main():
    MAX_P = 17                  # base tokens
    N_TOTAL_TASKS = len(ALL_TASKS)
    FIXED_VOCAB_SIZE = MAX_P + N_TOTAL_TASKS  # enough for all separator tokens

    out_dir = Path("src/configs/generated")
    out_dir.mkdir(parents=True, exist_ok=True)

    task_subsets = all_task_subsets(ALL_TASKS)
    

    print(f"Launching {len(task_subsets)} runs...\n")

    for subset in task_subsets:
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["data"]["tasks"] = list(subset)

        # Clean, readable run name
        task_names = "_".join(
            t["name"].replace("MovingWindow", "") for t in subset
        )
        run_name = f"mix_{task_names}"

        cfg["train"]["wandb_run_name"] = run_name
        cfg["model"]["vocab_size"] = FIXED_VOCAB_SIZE

        cfg_path = out_dir / f"{run_name}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        print(f"▶ Running {run_name}")
        subprocess.run(
            ["python", "train.py", "--config", str(cfg_path)],
            check=True,
        )

    print("\n✅ Sweep complete.")


if __name__ == "__main__":
    main()
