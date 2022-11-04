import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LinearLR
from torch.nn import BCELoss
import json

import pytorch_lightning as pl
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
import json 
import pandas as pd
import os
from .mbart_qg import MBARTQG


LANGUAGE_MAP = {
    "fr" : "[fr_XX]",
    "en" : "[en_XX]",

}

class MBARTHEZQGDataLoaderCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch_list: list) -> dict:
        context = [b['context'] for b in batch_list]
        question = [b['question'] for b in batch_list]
        input_lang = [b['input_lang'] for b in batch_list]
        output_lang = [b['output_lang'] for b in batch_list]

        source = self.tokenizer(context, return_tensors="pt",  padding='longest', truncation=True, max_length=512)
        target = self.tokenizer(question, return_tensors="pt",  padding='longest', truncation=True, max_length=512)

        source_input_ids = source.input_ids
        target_input_ids = target.input_ids
#         No source language input for MBarthez
        source_input_ids[:, 0] = torch.LongTensor(input_lang)
        target_input_ids[:, 0] = torch.LongTensor(output_lang)
        
        return {
            "input_ids": source_input_ids,
            "attention_mask": source.attention_mask,
            "labels": target_input_ids
        }

class MBARTHEZQG(MBARTQG):


    def __init__(
            self,
            pretrained_name = "moussaKam/mbarthez",
            fixed_encoder = False,
            validation_callback = None, 
            log_dir = None,
            optimizer = "adamw",
            learning_rate = 1e-4
        ):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.tokenizer.add_tokens(['<hl>', '[fr_XX]', '[en_XX]'], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def validation_step(self, batch, batch_idx):

        # validation loss
        output =\
            self.model(
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                labels = batch['labels']
            )
        loss = output.loss 

        # validation metrics based on generative approach
        with torch.no_grad():
            generated_batch = self.model.generate(
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids[LANGUAGE_MAP['fr']],
                max_new_tokens=200
            )
        generated_text = self.tokenizer.batch_decode(generated_batch, skip_special_tokens=True)
        ground_truth_text = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        self.log("val/loss",  loss.item(), reduce_fx="mean", sync_dist=True)
        return {"generated_text": generated_text, "ground_truth_text":ground_truth_text}
    
