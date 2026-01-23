import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import wandb  # only if config.train.wandb = True

# Optional, depending on your setup
import numpy as np

def calc_metric(t_train, t_test, model, prompt_len, gen_len, acc_start, device, k=None):
    # --- Forward pass and predictions ---
    attn_map, pre_lm_h, _, task_train_loss = model(
        t_train[:, :-1],
        targets=t_train[:, 1:],
        prompt_len=prompt_len,
        mask_input=True
    )

    t_train_pred = model.generate(idx=t_train[:, :prompt_len], max_new_tokens=gen_len, prompt_len=prompt_len)
    t_test_pred = model.generate(idx=t_test[:, :prompt_len], max_new_tokens=gen_len, prompt_len=prompt_len)

    # --- Accuracy ---
    t_train_acc = torch.mean((t_train_pred[:, acc_start:] == t_train[:, acc_start:]).float()).item()
    t_test_acc = torch.mean((t_test_pred[:, acc_start:] == t_test[:, acc_start:]).float()).item()

    # --- Input/output strings ---
    input_str = ", ".join(map(str, t_train[0].tolist()))
    output_str = ", ".join(map(str, t_train_pred[0].tolist()))

    # --- Per-token accuracy ---
    per_token_acc = {f"idx{i}_check": torch.mean(
        (t_train_pred[:, acc_start + i] == t_train[:, acc_start + i]).float()
    ).item() for i in range(gen_len)}

    # --- Pairwise cosine similarity ---
    embed_start = acc_start - 1
    embed_len = gen_len
    logit_cs = torch.zeros((embed_len, embed_len), device=device)
    for i_1 in range(embed_start, embed_start + embed_len):
        for i_2 in range(embed_start, i_1):
            logit_cs[i_1 - embed_start, i_2 - embed_start] = torch.mean(
                cosine_similarity(pre_lm_h[:, i_1, :], pre_lm_h[:, i_2, :], dim=-1), dim=0
            )

    per_pos_cos_sim = {f"mean_cosine_sim_{pos}": torch.sum(logit_cs[:, pos]) / (gen_len - 1 - pos)
                       for pos in range(gen_len-1)}
    mean_cosine_sim = torch.sum(logit_cs[:, 1:]) / (0.5 * (gen_len - 1) * (gen_len - 2))

    # --- Attention progress measure ---
    if k:
        attn_map_output_seq = attn_map[:, :, acc_start-1:]
        att_mask = torch.zeros_like(attn_map_output_seq, device=device)
        att_mask[:, :, 0, 0] = 1
        for i in range(gen_len-1):
            att_mask[:, :, i+1, i:i+k] = 1

        att_prog_measure = torch.mean(
            torch.sum(torch.abs(attn_map_output_seq) * att_mask, dim=(-3, -2, -1)) /
            torch.sum(torch.abs(attn_map_output_seq), dim=(-3, -2, -1)),
            dim=0
        )
    else:
        att_prog_measure = None

    # --- Cosine similarity figure ---
    logit_fig, ax_cs = plt.subplots(figsize=(10,10))
    im_cs = ax_cs.imshow(logit_cs.cpu())
    ax_cs.set_title("avg pre_lm_h cosine sim")
    logit_fig.colorbar(im_cs, ax=ax_cs, shrink=0.9)
    ax_cs.set_xticks(range(embed_len))
    ax_cs.set_yticks(range(embed_len))
    ax_cs.set_xlabel("Token index")
    ax_cs.set_ylabel("Token index")
    plt.tight_layout()
    for i1 in range(embed_len):
        for i2 in range(embed_len):
            ax_cs.text(i2, i1, f"{logit_cs[i1,i2].item():.2f}",
                       ha="center", va="center", color="w" if logit_cs[i1,i2] < logit_cs.max()/2 else "k")

    # --- Average attention per head figures ---
    avg_attn_per_head = attn_map.mean(dim=0).detach().cpu().numpy()
    n_heads = avg_attn_per_head.shape[0]
    attn_head_figs = {}
    for h in range(n_heads):
        fig_head, ax_head = plt.subplots(figsize=(10,10))
        im_head = ax_head.imshow(avg_attn_per_head[h])
        ax_head.set_title(f"Head {h} avg attention")
        fig_head.colorbar(im_head, ax=ax_head, shrink=0.9)
        ax_head.set_xticks(range(avg_attn_per_head[h].shape[-1]))
        ax_head.set_yticks(range(avg_attn_per_head[h].shape[-2]))
        attn_head_figs[f"head_{h}_avg_attention"] = fig_head

    return {
        "train_loss": task_train_loss.item(),
        "train_acc": t_train_acc,
        "test_acc": t_test_acc,
        "input_seq": input_str,
        "output_seq": output_str,
        "mean_cosine_sim": mean_cosine_sim.item(),
        "att_prog_measure": att_prog_measure.item() if att_prog_measure is not None else None,
        **per_token_acc,
        **per_pos_cos_sim,
        "cosine_sim_fig": logit_fig,
        "attn_head_figs": attn_head_figs
    }


def train_step(
    model,
    optim,
    data_samplers,
    step,
    config,
    device,
):
    num_tokens = config.data.num_tokens

    # --- MIXED BATCH SAMPLING ---
    mixed_train = {}
    mixed_test = {}

    ## Generate data for each task
    for name, task_info in data_samplers.items():
        sampler = task_info["sampler"]
        n_train = task_info["n_train"]
        n_test = task_info["n_test"]

        data = sampler.sample(
            num_samples=n_train + n_test,
            num_tokens=num_tokens,
        )
        train_part = data[:n_train, :]
        test_part = data[n_train:, :]
        mixed_train[name] = train_part
        mixed_test[name] = test_part

    ## Mix task batches together
    train_data = torch.cat(list(mixed_train.values()), dim=0)
    test_data = torch.cat(list(mixed_test.values()), dim=0)

    if config.data.mix == 'random':
        # optionally shuffle to fully mix across tasks
        perm = torch.randperm(train_data.size(0))
        train_data = train_data[perm]
        perm = torch.randperm(test_data.size(0))
        test_data = test_data[perm]

    # Define lengths for calculation
    prompt_len = num_tokens + 1
    gen_len = num_tokens
    acc_start = num_tokens + 1

    # Train model on entire mixed batch
    model.train()
    optim.zero_grad(set_to_none=True)

    _, _, _, loss = model(
        train_data[:, :-1], 
        targets=train_data[:, 1:], 
        prompt_len=prompt_len, 
        mask_input=config.train.mask_input,
    )
    loss.backward()

    if config.train.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

    optim.step()

    model.eval()
    with torch.no_grad():
        ### Per task metrics
        examples_seen_per_task = {name: task_info["n_train"] * (step + 1) for name, task_info in data_samplers.items()}
        per_task_metrics = {}
        for task_name, task_info in data_samplers.items():
            t_train = mixed_train[task_name]
            t_test = mixed_test[task_name]
            # Calculate metrics for this task
            metrics = calc_metric(
                t_train,
                t_test,
                model,
                prompt_len,
                gen_len,
                acc_start,
                device,
                k=task_info["sampler"].k,
            )
            per_task_metrics[task_name] = metrics

            if config.train.wandb:
                # Log everything for this task
                log_dict = {
                    f"{task_name}/examples_seen": examples_seen_per_task[task_name],
                    **{f"{task_name}/{k}": v for k, v in metrics.items() if k not in ["cosine_sim_fig", "attn_head_figs"]}
                }
                wandb.log(log_dict, step=step)

                # Log figures separately and then close them
                wandb.log({f"{task_name}/cosine_sim_fig": metrics["cosine_sim_fig"]}, step=step)
                plt.close(metrics["cosine_sim_fig"])
                for h, fig in metrics["attn_head_figs"].items():
                    wandb.log({f"{task_name}/{h}": fig}, step=step)
                    plt.close(fig)

        ## Overall mixed batch metrics
        overall_metrics = calc_metric(train_data, test_data, model, prompt_len, gen_len, acc_start, device)
        if config.train.wandb:
            log_dict = {k: v for k, v in overall_metrics.items() if k not in ["cosine_sim_fig", "attn_head_figs"]}
            wandb.log(log_dict, step=step)

            wandb.log({"cosine_sim_fig": overall_metrics["cosine_sim_fig"]}, step=step)
            plt.close(overall_metrics["cosine_sim_fig"])
            for h, fig in overall_metrics["attn_head_figs"].items():
                wandb.log({h: fig}, step=step)
                plt.close(fig)

        print(
            f"Step {step} -- Train loss: {overall_metrics['train_loss']}, "
            f"Train Acc: {overall_metrics['train_acc']} Test Acc: {overall_metrics['test_acc']}"
        )

        plt.close()
        del (
            logit_fig,
            ax_cs,
            logit_cs,
            ax_head,
            fig_head

        )

        if config.train.save_ckpt:
            if (step == 0) or ((step + 1) % config.train.ckpt_freq == 0):
                model.train()
                torch.save(
                    {
                        "epoch": step,
                        "model": model.state_dict(),
                        "optim": optim.state_dict(),
                        "train_loss": overall_metrics['train_loss'],
                        "test_acc": overall_metrics['test_acc'],
                    },
                    "./mws_k2_l1_h1_a16_n16.tar",
                )
                print(f"saved state at epoch {step} to {f'./mws_k2_l1_h1_a16_n16.tar'}")

                if config.train.wandb:
                    model_wandb = wandb.Artifact(
                        f"model_step{step}", type="model"
                    )
                    model_wandb.add_file(f"./mws_k2_l1_h1_a16_n16.tar")
                    wandb.log_artifact(model_wandb)
                    print("model uploaded to wandb")