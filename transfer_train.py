"""
Transfer learning script: train a model on phase-1 tasks, then fine-tune on
phase-2 tasks, saving checkpoints and tracking gradients throughout.

Usage:
    python transfer_train.py
    python transfer_train.py --config src/configs/transfer_mws_to_mwp.yaml
"""

import os
import random
import yaml
import argparse
from dotmap import DotMap

import numpy as np
import torch
from torch.optim import Adam
import wandb

import sys
sys.path.append("./src")
import data
from model import GPTLinear, GPTSoftmax
from multi_task_train import train_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


def load_config(path: str) -> DotMap:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return DotMap(cfg)


def prepare_data_samplers(config, device: str) -> dict:
    samplers = {}
    for task_cfg in config.data.tasks:
        task_name = task_cfg.name
        class_name = getattr(task_cfg, "task", task_name)
        task_class = getattr(data, class_name)
        samplers[task_name] = {
            "sampler": task_class(task_cfg, device),
            "n_train": task_cfg.n_train,
            "n_test": task_cfg.n_test,
        }
    return samplers


def save_checkpoint(model, optim, step: int, path: str, extra: dict = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        **(extra or {}),
    }
    torch.save(payload, path)
    print(f"Checkpoint saved at step {step} → {path}")


def run_phase(
    phase_name: str,
    model,
    optim,
    data_samplers: dict,
    config,
    device: str,
    global_step: int,
    ckpt_path: str,
) -> int:
    """Run one training phase. Returns the global step after the phase ends."""
    stop_on_perfect = config.train.get("stop_on_perfect_acc", False)
    patience = config.train.get("perfect_acc_patience", 25)
    perfect_counter = 0

    print(f"\n{'='*60}")
    print(f"  Starting {phase_name}  (tasks: {list(data_samplers.keys())})")
    print(f"{'='*60}\n")

    for step in range(config.train.num_steps):
        overall_metrics = train_step(
            model=model,
            optim=optim,
            data_samplers=data_samplers,
            step=global_step,
            config=config,
            device=device,
        )

        if config.train.save_ckpt and ((global_step + 1) % config.train.ckpt_freq == 0):
            base, ext = os.path.splitext(ckpt_path)
            step_ckpt_path = f"{base}_step{global_step:05d}{ext}"
            save_checkpoint(
                model, optim, global_step, step_ckpt_path,
                extra={"phase": phase_name, "test_acc": overall_metrics["test_acc"]},
            )
            if config.train.wandb:
                artifact = wandb.Artifact(f"{phase_name}_step{global_step}", type="model")
                artifact.add_file(step_ckpt_path)
                wandb.log_artifact(artifact)

        global_step += 1

        if stop_on_perfect:
            if overall_metrics["test_acc"] >= 1.0:
                perfect_counter += 1
            else:
                perfect_counter = 0
            if perfect_counter >= patience:
                print(
                    f"\n✅ Early stop ({phase_name}) at global step {global_step - 1}: "
                    f"100% accuracy held for {patience} consecutive steps\n"
                )
                break

    # Always save a final checkpoint at phase end
    base, ext = os.path.splitext(ckpt_path)
    final_ckpt_path = f"{base}_final{ext}"
    save_checkpoint(
        model, optim, global_step - 1, final_ckpt_path,
        extra={"phase": phase_name, "test_acc": overall_metrics["test_acc"]},
    )
    if config.train.wandb:
        artifact = wandb.Artifact(f"{phase_name}_final", type="model")
        artifact.add_file(final_ckpt_path)
        wandb.log_artifact(artifact)

    return global_step


# ---------------------------------------------------------------------------
# Weight comparison
# ---------------------------------------------------------------------------

def _load_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    for key in ("model", "model_state_dict", "state_dict"):
        if key in ckpt:
            return ckpt[key]
    # bare state dict
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise ValueError(f"Cannot find state dict in {path}")


def compare_checkpoints(path1: str, path2: str, wandb_enabled: bool = False):
    """Compare two checkpoints layer-by-layer and optionally log to W&B."""
    sd1 = _load_state_dict(path1)
    sd2 = _load_state_dict(path2)

    shared = sorted(set(sd1) & set(sd2))
    print(f"\n=== Weight Comparison: {os.path.basename(path1)} vs {os.path.basename(path2)} ===")
    print(f"Comparing {len(shared)} shared layers\n")

    rows = []
    for k in shared:
        w1, w2 = sd1[k], sd2[k]
        if w1.shape != w2.shape:
            print(f"Shape mismatch for {k}: {w1.shape} vs {w2.shape}")
            continue

        diff = w1 - w2
        abs_diff = diff.abs()
        l2 = diff.norm(2).item()
        cos = torch.nn.functional.cosine_similarity(
            w1.flatten().unsqueeze(0), w2.flatten().unsqueeze(0)
        ).item()

        rows.append({
            "layer": k,
            "l2_diff": l2,
            "mean_abs_diff": abs_diff.mean().item(),
            "max_abs_diff": abs_diff.max().item(),
            "cosine_sim": cos,
        })

    rows.sort(key=lambda r: r["mean_abs_diff"], reverse=True)

    for r in rows:
        print(
            f"{r['layer']:<50}  l2={r['l2_diff']:.3e}  "
            f"mean_abs={r['mean_abs_diff']:.3e}  cos_sim={r['cosine_sim']:.4f}"
        )

    if wandb_enabled:
        table = wandb.Table(
            columns=["layer", "l2_diff", "mean_abs_diff", "max_abs_diff", "cosine_sim"]
        )
        for r in rows:
            table.add_data(r["layer"], r["l2_diff"], r["mean_abs_diff"], r["max_abs_diff"], r["cosine_sim"])
        wandb.log({"weight_comparison": table})
        print("\nWeight comparison table logged to W&B.")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config.train.get("seed", 42))

    # Vocab / block size
    if not config.model.get("vocab_size", None):
        max_sep = max(t.sep for t in config.data.tasks)
        config.model.vocab_size = max(config.data.get("p", 17), config.data.get("max_num", 16), max_sep) + 1
    config.model.block_size = 2 * config.data.num_tokens + 1

    # Model
    model_cls = GPTLinear if config.model.linear else GPTSoftmax
    model = model_cls(config.model, return_att=True).to(device)

    if config.train.freeze_embedding:
        for param in model.transformer.wte.parameters():
            param.requires_grad = False
        for param in model.transformer.wpe.parameters():
            param.requires_grad = False

    optim = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.train.lr)

    # W&B
    if config.train.wandb:
        wandb.login(key="")
        wandb.init(
            project=config.train.wandb_project,
            name=config.train.get("wandb_run_name", None),
            config=config.toDict(),
            save_code=False,
        )
        watch_log_freq = config.train.get("watch_log_freq", 10)
        wandb.watch(model, log="all", log_freq=watch_log_freq)

    # Data samplers
    all_samplers = prepare_data_samplers(config, device)
    phase1_names = config.transfer.phase1_tasks
    phase2_names = config.transfer.phase2_tasks
    phase1_samplers = {k: all_samplers[k] for k in phase1_names}
    phase2_samplers = {k: all_samplers[k] for k in phase2_names}

    ckpt_phase1 = config.train.ckpt_path_phase1
    ckpt_phase2 = config.train.ckpt_path_phase2

    # Phase 1 — skip if final checkpoint already exists
    base, ext = os.path.splitext(ckpt_phase1)
    ckpt_phase1_final = f"{base}_final{ext}"
    if os.path.exists(ckpt_phase1_final):
        print(f"\nPhase 1 final checkpoint found at {ckpt_phase1_final}, loading and skipping phase 1.\n")
        ckpt = torch.load(ckpt_phase1_final, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        global_step = ckpt["step"] + 1
    else:
        global_step = run_phase(
            phase_name="phase1",
            model=model,
            optim=optim,
            data_samplers=phase1_samplers,
            config=config,
            device=device,
            global_step=0,
            ckpt_path=ckpt_phase1,
        )

    # Optionally freeze embeddings for phase 2
    if config.transfer.get("freeze_embeddings_phase2", False):
        for param in model.transformer.wte.parameters():
            param.requires_grad = False
        for param in model.transformer.wpe.parameters():
            param.requires_grad = False
        print("Embeddings frozen for phase 2.")
        # Rebuild optimizer to exclude frozen params
        optim = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.train.lr)

    # Phase 2
    run_phase(
        phase_name="phase2",
        model=model,
        optim=optim,
        data_samplers=phase2_samplers,
        config=config,
        device=device,
        global_step=global_step,
        ckpt_path=ckpt_phase2,
    )

    # Weight comparison
    if config.transfer.get("compare_weights", True):
        base2, ext2 = os.path.splitext(ckpt_phase2)
        ckpt_phase2_final = f"{base2}_final{ext2}"
        compare_checkpoints(ckpt_phase1_final, ckpt_phase2_final, wandb_enabled=config.train.wandb)

    if config.train.wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer learning between tasks")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/transfer_mws_to_mwp.yaml",
        help="Path to YAML config file",
    )
    main(parser.parse_args())
