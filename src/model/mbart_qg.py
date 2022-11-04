import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LinearLR
from torch.nn import BCELoss
import json

import pytorch_lightning as pl
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from .optim_utils import OPTIM_MAP
import json 
import pandas as pd
import os

LANGUAGE_MAP = {
    "fr" : "fr_XX",
    "en" : "en_XX",
    "es" : "es_XX",
    "ja" : "ja_XX",
    "pt" : "pt_XX",
    "tl" : "tl_XX"
}

class MBARTQGDataLoaderCollator:
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

        source_input_ids[:, 0] = torch.LongTensor(input_lang)
        target_input_ids[:, 0] = torch.LongTensor(output_lang)
        
        return {
            "input_ids": source_input_ids,
            "attention_mask": source.attention_mask,
            "labels": target_input_ids
        }

class MBARTQG(pl.LightningModule):


    def __init__(
            self,
            pretrained_name = "facebook/mbart-large-50-many-to-many-mmt",
            fixed_encoder = False,
            validation_callback = None, 
            log_dir = None,
            optimizer = "adamw",
            learning_rate = 1e-4
        ):
        super().__init__()
        self.fixed_encoder = fixed_encoder
        self.model = MBartForConditionalGeneration.from_pretrained(pretrained_name)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(pretrained_name)
        self.tokenizer.add_tokens(['<hl>'], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.validation_callback = validation_callback
        self.log_dir = log_dir
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
    
    def training_step(self, batch, batch_idx):
        output =\
            self.model(
                **batch
            )
        loss = output.loss 
        self.log("training_reconstruction_loss",  output.loss.item(), reduce_fx="mean")
        return loss
    
    def configure_optimizers(self):
        optimizable_parameters = list(self.model.model.decoder.parameters()) + list(self.model.lm_head.parameters()) 
        if(not self.fixed_encoder):
            optimizable_parameters = self.model.parameters()
        optimizer = OPTIM_MAP[self.optimizer_name](optimizable_parameters, lr=self.learning_rate)
        scheduler = {
            "scheduler": LinearLR(optimizer, total_iters = 100, start_factor= 1.0 / 1000.),
            "interval": "step",
            'name': 'lr_scheduler',
            "frequency": 1
        }
        return [optimizer], [scheduler]

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
                forced_bos_token_id=self.tokenizer.lang_code_to_id[LANGUAGE_MAP['fr']],
                max_new_tokens=200
            )
        generated_text = self.tokenizer.batch_decode(generated_batch, skip_special_tokens=True)
        ground_truth_text = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        self.log("val/loss",  loss.item(), reduce_fx="mean", sync_dist=True)
        return {"generated_text": generated_text, "ground_truth_text":ground_truth_text}
    
    def validation_epoch_end(self, batch, *kargs, **kwargs):
        predictions = sum([b["generated_text"] for b in batch], [])
        references = sum([b["ground_truth_text"] for b in batch], [])
        if self.validation_callback is not None:
            validation_log =  self.validation_callback(predictions, references)
            for k, v in validation_log.items():
                self.log("val/"+k, v)
        if self.log_dir != None:
            df = pd.DataFrame({"predictions": predictions, "references": references})
            df.to_csv(os.path.join(self.log_dir, "validation_prediction-"+str(self.current_epoch)+".csv"))

    def on_after_backward(self, *kargs, **kwargs):
        self.model.model.shared.weight._grad[:-1] = 0