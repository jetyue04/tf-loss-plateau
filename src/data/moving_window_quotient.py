import torch
from .base_data import BaseDataTask


class MovingWindowQuotient(BaseDataTask):
    def __init__(self, config, device="cuda"):
        super().__init__(config, device)
        self.min_num = config.min_num
        self.max_num = config.max_num
        self.k = config.k
        self.p = config.p
        self.sep = config.sep
        assert self.p > self.max_num

    @torch.no_grad()
    def sample(self, num_samples, num_tokens):
        # Sample integers
        x = torch.randint(
            low=self.min_num,
            high=self.max_num + 1,
            size=(num_samples, num_tokens),
            device=self.device,
        )

        # Outgoing / incoming terms
        x_out = x[:, : num_tokens - self.k + 1]     # denominator
        x_in  = x[:, self.k - 1 :]                   # numerator

        # Modular inverse via Fermat
        x_out_inv = torch.pow(x_out, self.p - 2) % self.p

        # Modular quotient
        q = (x_in * x_out_inv) % self.p              # (B, T - k + 1)

        # Align into full-length tensor
        q_full = torch.zeros_like(x)
        q_full[:, self.k - 1 :] = q

        # Final concatenated sample
        samples = torch.cat(
            [
                x,
                self.sep * torch.ones((num_samples, 1), device=self.device),
                q_full,
            ],
            dim=-1,
        ).to(torch.int64)

        return samples
