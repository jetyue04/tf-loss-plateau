import torch
import numpy as np
from .base_data import BaseDataTask

class MovingWindowProduct(BaseDataTask):
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
        
        moving_product = random_ints.clone().detach()
        for j in range(self.k - 1, random_ints.shape[1]):
            # product over the window ending at index j
            moving_product[:, j] = torch.prod(random_ints[:, j-self.k+1:j+1], dim=1)
        samples = (
            torch.cat(
                [
                    random_ints,
                    self.sep * torch.ones(size=(num_samples, 1)).to(self.device),
                    torch.remainder(input=moving_product, other=self.p),
                ],
                axis=-1,
            )
            .to(int)
            .detach()
        )

        return samples