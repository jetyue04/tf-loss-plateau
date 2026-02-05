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
]

# -------------------------------------------------------------------
# Base config template
# -------------------------------------------------------------------
BASE_CONFIG = {
    "model": {
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 256,
        "linear": True,
    },
    "data": {
        "tasks": [],
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
        "num_steps": 1000,
        "norm_type": "none_rank",
        "wandb": True,
        "wandb_project": "loss_plateau_tf",
        "save_ckpt": False,
        "ckpt_freq": 20,
        "seed": 67,
        "mask_input": True,
        "freeze_embedding": False,
        "stop_on_perfect_acc": True,
        "perfect_acc_patience": 50,
        "perfect_acc_eps": 1e-6,
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
    TOTAL_TRAIN_EXAMPLES = 128   # total across all tasks
    TOTAL_TEST_EXAMPLES = 64     # total across all tasks
    MAX_P = 17
    N_TOTAL_TASKS = len(ALL_TASKS)
    FIXED_VOCAB_SIZE = MAX_P + N_TOTAL_TASKS

    out_dir = Path("src/configs/generated")
    out_dir.mkdir(parents=True, exist_ok=True)

    task_subsets = all_task_subsets(ALL_TASKS)
    
    print(f"Launching {len(task_subsets)} runs...\n")

    for subset in task_subsets:
        cfg = copy.deepcopy(BASE_CONFIG)
        n_tasks = len(subset)

        # Evenly split train/test across tasks
        per_task_train = TOTAL_TRAIN_EXAMPLES // n_tasks
        per_task_test = TOTAL_TEST_EXAMPLES // n_tasks

        # Assign remainder to last task if not divisible
        train_remainder = TOTAL_TRAIN_EXAMPLES - per_task_train * n_tasks
        test_remainder = TOTAL_TEST_EXAMPLES - per_task_test * n_tasks

        tasks = []
        for i, t in enumerate(subset):
            t_copy = copy.deepcopy(t)
            t_copy["n_train"] = per_task_train
            t_copy["n_test"] = per_task_test
            if i == n_tasks - 1:
                t_copy["n_train"] += train_remainder
                t_copy["n_test"] += test_remainder
            tasks.append(t_copy)

        cfg["data"]["tasks"] = tasks

        # Clean, readable run name
        task_names = "_".join(
            t["name"].replace("MovingWindow", "") for t in subset
        )
        run_name = f"mix_{task_names}_freezeEmbedding{cfg['train']['freeze_embedding']}"

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
