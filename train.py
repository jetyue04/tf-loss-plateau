import math
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
from model_linear import GPTLinear
from model_softmax import GPTSoftmax
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


def prepare_data_samplers(config):
    """Create a dict of data samplers for each task."""
    num_task = len(config.data.tasks)
    data_samplers = {}
    for i in range(num_task):
        task = config.data.tasks[i]
        task_class = getattr(data, task.name)
        data_samplers[task.name] = task_class(
            min_num=config.data.min_num,
            max_num=config.data.max_num,
            k=config.data.k if hasattr(config.data, 'k') else None,
            p=config.data.p if hasattr(config.data, 'p') else None,
            sep=task.sep,
        )
    return data_samplers


def main(args):
    # Load config
    config = load_config(args.config)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seeds
    seed = getattr(config.train, "seed", 42)
    set_seed(seed)

    # Model settings -- May needs tweaking
    config.model.vocab_size = max(getattr(config.data, "p", 16), config.data.max_num) + 1
    config.model.block_size = 2 * config.data.num_tokens + 1

    # Prepare data samplers
    data_samplers = prepare_data_samplers(config)

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
        wandb_run_name = getattr(config.train, "wandb_run_name", None)
        wandb.login(key="")
        wandb.init(project=config.train.wandb_project, name=wandb_run_name, config=config)
        wandb.watch(model)

    # Training loop
    for step in range(config.train.num_steps):
        train_step(
            model=model,
            optim=optim,
            data_samplers=data_samplers,
            step=step,
            config=config,
            device=device,
        )

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
