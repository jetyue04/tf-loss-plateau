import math
import os
import random
import yaml
import argparse
from dotmap import DotMap

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# import matplotlib.pyplot as plt
import wandb

import sys
sys.path.append("./src")  # make sure Python can find src/
import data
from model import GPTLinear, GPTSoftmax
from multi_task_train import train_step


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ## Not sure if below would work if I dont have gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


def load_config(config_path: str):
    """Load YAML config and convert to DotMap."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = DotMap(cfg)
    return cfg


def prepare_data_samplers(config, device):
    """Create a dict of data samplers for each task."""
    num_task = len(config.data.tasks)
    data_samplers = {}
    for task_config in config.data.tasks:
        task_name = task_config.name        # unique identifier
        task_class_name = task_config.task  # the actual class to instantiate
        task_class = getattr(data, task_class_name)
        data_samplers[task_name] = {
            "sampler": task_class(task_config, device),
            "n_train": task_config.n_train,
            "n_test": task_config.n_test,
        }
    return data_samplers


def main(args):
    # Load config
    config = load_config(args.config)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seeds
    seed = config.train.get("seed", 42)
    set_seed(seed)

    # Model settings -- May needs tweaking
    n_tasks = len(config.data.tasks)
    
    # if config does not specify vocab size, calculate dynamically
    if not config.model.get("vocab_size", None):
        print("Calculating vocab size dynamically...")
        max_sep = max(task.sep for task in config.data.tasks)
        # config.model.vocab_size = max(getattr(config.data, "p", 17), config.data.max_num) + n_tasks
        config.model.vocab_size = max(getattr(config.data, "p", 17), config.data.max_num, max_sep) + 1

    config.model.block_size = 2 * config.data.num_tokens + 1

    # Create checkpoint directory if needed
    if getattr(config.train, "save_ckpt", False):
        ckpt_path = getattr(config.train, "ckpt_path", "./checkpoint.tar")
        ckpt_dir = os.path.dirname(ckpt_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

    # Prepare data samplers
    data_samplers = prepare_data_samplers(config, device)

    # Initialize model
    if config.model.linear:
        model = GPTLinear(config.model, return_att=True).to(device)
    else:
        model = GPTSoftmax(config.model, return_att=True).to(device)

    if config.train.freeze_embedding:
        for param in model.transformer.wte.parameters():
            param.requires_grad = False
        for param in model.transformer.wpe.parameters():
            param.requires_grad = False
    # Optimizer
    optim = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.train.lr)

    # WandB setup
    if getattr(config.train, "wandb", False):
        wandb_run_name = config.train.get("wandb_run_name", None)
        wandb.login(key="")
        wandb.init(project=config.train.wandb_project, name=wandb_run_name, config=config, save_code=False)
        watch_log_freq = config.train.get("watch_log_freq", 10)
        wandb.watch(model, log="all", log_freq=watch_log_freq)

    stop_on_perfect = config.train.get("stop_on_perfect_acc", False)
    perfect_patience = config.train.get("perfect_acc_patience", 50)
    # acc_eps = config.train.get("perfect_acc_eps", 1e-6)

    perfect_counter = 0

    # Training loop
    for step in range(config.train.num_steps):
        overall_metrics = train_step(
            model=model,
            optim=optim,
            data_samplers=data_samplers,
            step=step,
            config=config,
            device=device,
        )
        
        ## Early stop
        if stop_on_perfect:
            acc = overall_metrics["test_acc"]  # or "train_acc" if you prefer

            if acc >= 1.0:
                perfect_counter += 1
            else:
                perfect_counter = 0

            if perfect_counter >= perfect_patience:
                print(
                    f"\n✅ Early stopping at step {step}: "
                    f"overall accuracy stayed at 100% for "
                    f"{perfect_patience} consecutive steps\n"
                )
                break

    if getattr(config.train, "wandb", False):
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT model on multiple tasks")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/mix1_mws_mwp.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(args)