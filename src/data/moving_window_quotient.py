import torch
import numpy as np
from .base_data import BaseDataTask

class MovingWindowQuotient(BaseDataTask):
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
        # (B, T)
        x = torch.randint(
            low=self.min_num,
            high=self.max_num + 1,
            size=(num_samples, num_tokens),
            device=self.device,
        )

        # Output container
        out = torch.zeros_like(x)

        # Outgoing and incoming elements
        x_out = x[:, :-self.k]          # x_i
        x_in  = x[:, self.k:]           # x_{i+k}

        # Modular inverse of outgoing
        x_out_inv = torch.pow(x_out, self.p - 2) % self.p

        # Quotient
        q = (x_in * x_out_inv) % self.p

        # Align like moving window ops
        out[:, self.k:] = q

        return out