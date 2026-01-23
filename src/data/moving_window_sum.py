import torch
import numpy as np
from .base_data import BaseDataTask

class MovingWindowSum(BaseDataTask):
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
        random_ints = torch.randint(
            low=self.min_num, high=self.max_num + 1, size=(num_samples, num_tokens)
        ).to(self.device)

        random_ints_np = random_ints.detach().cpu().numpy()
        convolution = torch.stack(
            [
                torch.from_numpy(
                    np.convolve(
                        random_ints_np[i],
                        np.ones(self.k),
                        mode="valid",
                    )
                )
                for i in range(random_ints.shape[0])
            ]
        )
        moving_sum = random_ints.clone().detach()
        moving_sum[:, self.k - 1 :] = convolution

        samples = (
            torch.cat(
                [
                    random_ints,
                    self.sep * torch.ones(size=(num_samples, 1)).to(self.device),
                    torch.remainder(input=moving_sum, other=self.p),
                ],
                axis=-1,
            )
            .to(int)
            .detach()
        )
        return samples
