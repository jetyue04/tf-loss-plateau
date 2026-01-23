import torch
import numpy as np
from .base_data import BaseDataTask

class PrefixSum(BaseDataTask):
    def __init__(self, config, device="cuda"):
        super().__init__(config, device)
        self.min_num = getattr(config, "min_num", 1)
        self.no_repeat = getattr(config, "no_repeat", False)
        self.max_num = getattr(config, "max_num", 16)
        self.k = getattr(config, "k", 2)
        self.p = getattr(config, "p", 17)
        self.sep = getattr(config, "sep", 17)
        assert self.p > self.max_num
    @torch.no_grad()
    def sample(self, num_samples, num_tokens):
        if self.no_repeat:
            random_ints = torch.arange(start=self.min_num, end=self.max_num+1).view(-1, num_tokens).repeat(num_samples, 1).to(self.device)

            for i in range(num_samples):
                random_ints[i, :] = random_ints[i, torch.randperm(num_tokens)]

        else:
            random_ints = torch.randint(
                low=self.min_num, high=self.max_num + 1, size=(num_samples, num_tokens)
            ).to(self.device)

        prefix_mod_sum = torch.remainder(torch.cumsum(random_ints, dim=-1), self.p)

        samples = (
            torch.cat(
                [
                    random_ints,
                    self.p * torch.ones(size=(num_samples, 1)).to(self.device),
                    prefix_mod_sum,
                ],
                axis=-1,
            )
            .to(int)
            .detach()
        )

        return samples