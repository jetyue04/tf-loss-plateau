import torch
import numpy as np
from .base_data import BaseDataTask

class MovingWindowDifference(BaseDataTask):
    def __init__(self, config, device="cuda"):
        super().__init__(config, device)
        self.min_num = getattr(config, "min_num", 1)
        self.max_num = getattr(config, "max_num", 16)
        self.k = getattr(config, "k", 2)
        self.p = getattr(config, "p", 17)
        self.sep = getattr(config, "sep", 17)
        assert self.p > self.max_num

    @torch.no_grad()
    def sample(self, num_samples, num_tokens):
        # Generate random integers
        random_ints = torch.randint(
            low=self.min_num,
            high=self.max_num + 1,
            size=(num_samples, num_tokens),
            device=self.device,
        )

        # Initialize moving_difference
        moving_difference = random_ints.clone().float()  # use float for safety

        # Compute moving difference
        for i in range(num_tokens):
            if i < self.k:
                # Growing prefix difference
                window = random_ints[:, :i+1]
                diff = window[:, 0].clone()
                if i > 0:
                    diff -= torch.sum(window[:, 1:], dim=1)
                moving_difference[:, i] = diff
            else:
                # Fixed-size window difference
                window = random_ints[:, i-self.k+1:i+1]
                diff = window[:, 0].clone()
                diff -= torch.sum(window[:, 1:], dim=1)
                moving_difference[:, i] = diff

        # Concatenate input, separator, and modulo
        samples = torch.cat(
            [
                random_ints,
                self.sep * torch.ones((num_samples, 1), device=self.device),
                torch.remainder(moving_difference, self.p),
            ],
            dim=-1
        ).to(int)

        return samples
    # def sample(self, num_samples, num_tokens):
    #     random_ints = torch.randint(
    #         low=self.min_num, high=self.max_num + 1, size=(num_samples, num_tokens)
    #     ).to(self.device)

    #     random_ints_np = random_ints.detach().cpu().numpy()

    #     moving_difference = random_ints.clone().detach()
    #     moving_difference = random_ints.clone()

    #     for j in range(self.k - 1, num_tokens):
    #         window = random_ints[:, j - self.k + 1 : j + 1]  # shape (num_samples, k)
    #         d = window[:, 0]
    #         for t in range(1, self.k):
    #             d = d - window[:, t]  # subtract all other elements in the window
    #         moving_difference[:, j] = d

    #     samples = (
    #         torch.cat(
    #             [
    #                 random_ints,
    #                 self.sep * torch.ones(size=(num_samples, 1)).to(self.device),
    #                 torch.remainder(input=moving_difference, other=self.p),
    #             ],
    #             axis=-1,
    #         )
    #         .to(int)
    #         .detach()
    #     )

    #     return samples