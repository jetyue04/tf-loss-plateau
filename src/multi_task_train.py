import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import wandb  # only if config.train.wandb = True

# Optional, depending on your setup
import numpy as np

def train_step(
    model,
    optim,
    data_samplers,
    step,
    config,
    device,
):
    n_train, n_test, num_tokens = (
        config.data.n_train,
        config.data.n_test,
        config.data.num_tokens,
    )

    # --- MIXED BATCH SAMPLING ---
    # task_names = list(data_samplers.keys())
    n_tasks = len(data_samplers)

    n_train_each = n_train // n_tasks
    n_test_each = n_test // n_tasks

    mixed_train = {}
    mixed_test = {}

    for name, sampler in data_samplers.items():
        data = sampler.sample(
            num_samples=n_train_each + n_test_each,
            num_tokens=num_tokens,
        )
        train_part = data[:n_train_each, :]
        test_part = data[n_train_each:, :]
        mixed_train[name] = train_part
        mixed_test[name] = test_part

    train_data = torch.cat(list(mixed_train.values()), dim=0)
    test_data = torch.cat(list(mixed_test.values()), dim=0)

    if config.data.mix == 'random':
        # optionally shuffle to fully mix across tasks
        perm = torch.randperm(train_data.size(0))
        train_data = train_data[perm]
        perm = torch.randperm(test_data.size(0))
        test_data = test_data[perm]

    prompt_len = num_tokens + 1
    gen_len = num_tokens
    acc_start = num_tokens + 1

    model.train()
    optim.zero_grad(set_to_none=True)

    _, _, _, loss = model(
        train_data[:, :-1], 
        targets=train_data[:, 1:], 
        prompt_len =prompt_len, 
        mask_input=config.train.mask_input,
    )
    loss.backward()

    if config.train.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

    optim.step()

    model.eval()
    with torch.no_grad():
        ### Per task metrics
        # if n_tasks > 1:
        examples_seen_per_task = n_train_each * (step + 1)
        per_task_metrics = {}

        for task_name in mixed_train.keys():
            t_train = mixed_train[task_name]
            t_test = mixed_test[task_name]

            # Compute loss on this task
            _, _, _, task_train_loss = model(
                t_train[:, :-1],
                targets=t_train[:, 1:],
                prompt_len=prompt_len,
                mask_input=config.train.mask_input,
            )

            # Predictions
            t_train_pred = model.generate(
                idx=t_train[:, :prompt_len],
                max_new_tokens=gen_len,
                prompt_len=prompt_len,
            )
            t_test_pred = model.generate(
                idx=t_test[:, :prompt_len],
                max_new_tokens=gen_len,
                prompt_len=prompt_len,
            )

            # Accuracy for this task
            t_train_acc = torch.mean(
                (t_train_pred[:, acc_start:] == t_train[:, acc_start:]).float()
            ).item()
            t_test_acc = torch.mean(
                (t_test_pred[:, acc_start:] == t_test[:, acc_start:]).float()
            ).item()

            # pick the first sequence in the batch
            input_seq = t_train[0].tolist() 
            pred_seq = t_train_pred[0].tolist()
            # Convert to string for logging
            input_str = ", ".join(map(str, input_seq))
            pred_str = ", ".join(map(str, pred_seq))

            # if config.train.wandb:
            #     wandb.log({
            #         f"{task_name}_input_seq": input_str,
            #         f"{task_name}_pred_seq": pred_str,
            #     }, step=step)


            per_task_metrics[task_name] = {
                "train_loss": task_train_loss.item(),
                "train_acc": t_train_acc,
                "test_acc": t_test_acc,
                "input_seq": input_str,
                "output_seq": pred_str,
                # "num_train_samples": t_train.shape[0],
                # "num_test_samples": t_test.shape[0],
            }
            
        if config.train.wandb:
            for task_name, vals in per_task_metrics.items():
                wandb.log({
                    f"{task_name}/train_loss": vals["train_loss"],
                    f"{task_name}/train_acc": vals["train_acc"],
                    f"{task_name}/test_acc": vals["test_acc"],
                    f"{task_name}/input_seq": vals['input_seq'],
                    f"{task_name}/output_seq": vals['output_seq'],
                    # f"{task_name}/num_train_samples": vals["num_train_samples"],
                    f"{task_name}/examples_seen": examples_seen_per_task,
                }, step=step)

        # Log train loss, train / test acc, repetition frequency
        attn_map, pre_lm_h, _, train_loss = model(
            train_data[:, :-1], 
            targets=train_data[:, 1:], 
            prompt_len =prompt_len, 
            mask_input=config.train.mask_input,
            )

        train_pred = model.generate(
            idx=train_data[:, :prompt_len],
            max_new_tokens=gen_len,
            prompt_len =prompt_len,
        )
        test_pred = model.generate(
            idx=test_data[:, :prompt_len],
            max_new_tokens=gen_len,
            prompt_len =prompt_len,
        )

        train_acc = torch.mean(
            (train_pred[:, acc_start:] == train_data[:, acc_start:]).to(float)
        ).item()
        test_acc = torch.mean(
            (test_pred[:, acc_start:] == test_data[:, acc_start:]).to(float)
        ).item()

        data_repeat_frac = torch.mean((test_data[:, acc_start:-1] == test_data[:, acc_start+1:]).to(float))
        model_repeat_frac = torch.mean((test_pred[:, acc_start:-1] == test_pred[:, acc_start+1:]).to(float))

        # Log attention progress measure
        attn_map_output_seq = attn_map[:, :, acc_start-1:]
        att_mask = torch.zeros_like(attn_map_output_seq).to(device)

        att_mask[:, :, 0, 0] = 1
        for i in range(num_tokens - 1):
            att_mask[:, :, i + 1, i : i + 2] = 1

        att_prog_measure = torch.mean(
            torch.sum(torch.abs(attn_map_output_seq) * att_mask, dim=(-3, -2, -1)) /
            torch.sum(torch.abs(attn_map_output_seq), dim=(-3, -2, -1)),
            dim=0
        )

        # Log pair-wise cosine similarity between hidden states
        embed_start = acc_start - 1
        embed_len = gen_len

        logit_cs = torch.zeros((embed_len, embed_len))

        for i_1 in range(embed_start, embed_start + embed_len):
            for i_2 in range(embed_start, i_1):
                logit_cs[i_1 - embed_start, i_2 - embed_start] = torch.mean(
                    (
                        cosine_similarity(
                            pre_lm_h[:, i_1, :], pre_lm_h[:, i_2, :], dim=-1
                        )
                    ), dim=0
                )

        # --- Cosine similarity figure ---
        logit_fig, ax_cs = plt.subplots(figsize=(10,10))
        im_cs = ax_cs.imshow(logit_cs)
        ax_cs.set_title("avg pre_lm_h cosine sim")
        logit_fig.colorbar(im_cs, ax=ax_cs, shrink=0.9)
        ax_cs.set_xticks(range(embed_len))
        ax_cs.set_yticks(range(embed_len))
        ax_cs.set_xlabel("Token index")
        ax_cs.set_ylabel("Token index")
        plt.tight_layout()
        # plt.show()

        # Optional: overlay numbers
        for i1 in range(embed_len):
            for i2 in range(embed_len):
                ax_cs.text(i2, i1, f"{logit_cs[i1, i2].item():.2f}",
                        ha="center", va="center", color="w" if logit_cs[i1, i2] < logit_cs.max()/2 else "k")
    

        if config.train.wandb:
            wandb.log({"pre_lm_h_cosine_sim": logit_fig}, step=step)
        plt.close(logit_fig)


        # --- Attention maps per head (averaged over batch) ---
        avg_attn_per_head = attn_map.mean(dim=0).detach().cpu().numpy()  # shape: (n_head, T, T)

        for h in range(config.model.n_head):
            fig_head, ax_head = plt.subplots(figsize=(10,10))
            im_head = ax_head.imshow(avg_attn_per_head[h])
            ax_head.set_title(f"Head {h} avg attention")
            fig_head.colorbar(im_head, ax=ax_head, shrink=0.9)
            ax_head.set_xticks(range(avg_attn_per_head[h].shape[-1]))
            ax_head.set_yticks(range(avg_attn_per_head[h].shape[-2]))
            
            if config.train.wandb:
                wandb.log({f"Head_{h}_avg_attention": fig_head}, step=step)
            plt.close(fig_head)


        print(
            f"Step {step} -- Train loss: {train_loss}, Train Acc: {train_acc} Test Acc: {test_acc}"
        )
        # print(f"input: {test_data[0]} \n predicted:{test_pred[0]}")
        if config.train.wandb:
            
            log_data = {
                'examples_seen_per_task': examples_seen_per_task,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "data_repeat_frac": data_repeat_frac,
                "model_repeat_frac": model_repeat_frac,
                "att_prog_measure": att_prog_measure,
                # "pre_lm_h_cosine_sim": logit_fig,
                "mean_cosine_sim": torch.sum(logit_cs[:, 1:]) / (0.5 * (gen_len-1) * (gen_len-2))
            }

            for output_pos in range(gen_len):
                log_data.update(
                    {
                        f"idx{output_pos}_check": torch.mean(
                            (train_pred[:, acc_start + output_pos] == train_data[:, acc_start + output_pos]).to(float)
                        ).item()
                    }
                )

                if output_pos < gen_len-1:
                    log_data.update(
                        {
                            f"mean_cosine_sim_{output_pos}": torch.sum(logit_cs[:, output_pos]) / (gen_len-1-output_pos)
                        }
                    )

            wandb.log(log_data, step=step)

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
                        "train_loss": train_loss,
                        "test_acc": test_acc,
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