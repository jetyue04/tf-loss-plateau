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

        # Precompute modular inverses (0 unused)
        self.inv_table = torch.zeros(self.p, dtype=torch.long, device=self.device)
        for i in range(1, self.p):
            self.inv_table[i] = pow(i, -1, self.p)

    @torch.no_grad()
    def sample(self, num_samples, num_tokens):
        x = torch.randint(
            low=self.min_num,
            high=self.max_num + 1,
            size=(num_samples, num_tokens),
            device=self.device,
        )

        B, T = x.shape
        q_full = torch.zeros_like(x)

        # y0 = x0
        q_full[:, 0] = x[:, 0]

        for j in range(1, T):
            # previous k-1 elements
            start = max(0, j - (self.k - 1))
            window = x[:, start:j]

            # product of previous window
            denom = torch.prod(window, dim=1) % self.p

            # modular inverse
            denom_inv = self.inv_table[denom]

            # modular division
            q_full[:, j] = (x[:, j] * denom_inv) % self.p

        samples = torch.cat(
            [
                x,
                self.sep * torch.ones((num_samples, 1), device=self.device),
                q_full,
            ],
            dim=-1,
        ).to(torch.int64)

        return samples
