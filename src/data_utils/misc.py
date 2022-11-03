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


class CollateTokenizer():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch : dict) -> dict:
        print(batch)
        source = self.tokenizer(batch['src'], return_tensors="pt",  padding='longest', truncation=True, max_length=512)
        target = self.tokenizer(batch['tgt'], return_tensors="pt",  padding='longest', truncation=True, max_length=512)

        source_input_ids = source.input_ids
        target_input_ids = target.input_ids

        source_input_ids[:, 0] = batch["input_lang"]
        target_input_ids[:, 0] = batch["output_lang"]
        
        return {
            "input_ids": source_input_ids,
            "attention_mask": source.attention_mask,
            "labels": target_input_ids
        }
