from typing import TypeVar, Generic, Sequence

import torch

class Padding():
    def __init__(self, padding_value = 0, dim=-1):
        self.padding_value = padding_value
        self.dim = dim
    
    def __call__(self, tensor: torch.Tensor, size : int) -> torch.Tensor:
        pad_size = list(tensor.shape)
        pad_size[self.dim] = size - tensor.size(self.dim)
        return torch.cat([tensor, tensor.new(*pad_size).fill_(self.padding_value)], dim=self.dim)
    
class CollatePadding():
    def __init__(self, padding_value = 0, padding_dim = -1, stack_dim = 0):
        self.padding_dim = padding_dim
        self.stack_dim = stack_dim
        self.padder = Padding(padding_value, padding_dim)
    
    def __call__(self, batch : Sequence[torch.Tensor]) -> torch.Tensor:
        max_len = max(map(lambda x: x.shape[self.padding_dim], batch))
        tr_batch = [self.padder(x, size=max_len) for x in  batch]
        return torch.cat(tr_batch, dim=self.stack_dim)


