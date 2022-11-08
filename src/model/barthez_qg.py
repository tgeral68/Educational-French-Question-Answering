import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import json

import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import get_scheduler

import os
import pandas as pd


class DataCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        src_txt = [sample['context'] for sample in batch]
        tgt_txt = [sample['question'] for sample in batch]
        src_tok = self.tokenizer(src_txt, return_tensors="pt",  padding='longest', truncation=True, max_length=512)
        tgt_tok = self.tokenizer(tgt_txt, return_tensors="pt",  padding='longest', truncation=True, max_length=512).input_ids

        return {
            **src_tok,
            "labels": tgt_tok
        }

class BarthezQA(pl.LightningModule):

    def __init__(
        self,
        model_name = "moussaKam/barthez",
        load_pretraned_model = False,
        validation_callback = None, 
        log_dir = None
        ):

        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_tokens(['<hl>'], special_tokens=True) # ajouter <hl>

        if load_pretraned_model != False:
            self.model = torch.load(load_pretraned_model)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.validation_callback = validation_callback
        self.log_dir = log_dir
    
    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss
        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=1e-4)
        scheduler = {
            "scheduler": LinearLR(optimizer, total_iters = 1000, start_factor= 1.0 / 1000.),
            "interval": "step",
            'name': 'lr_scheduler',
            "frequency": 1
        }
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss
        with torch.no_grad():
            generated_batch = self.model.generate(
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'])
        generated_text = self.tokenizer.batch_decode(generated_batch, skip_special_tokens=True)
        ground_truth_text = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        self.log("val_loss", loss.item())
        return {"generated_text": generated_text, "ground_truth_text":ground_truth_text}
    
    def validation_epoch_end(self, batch, *kargs, **kwargs):
        predictions = sum([b["generated_text"] for b in batch], [])
        references = sum([b["ground_truth_text"] for b in batch], [])
        if self.validation_callback is not None:
            validation_log =  self.validation_callback(predictions, references)
            for k, v in validation_log.items():
                self.log("val_"+k, v)
        if self.log_dir != None:
            df = pd.DataFrame({"predictions": predictions, "references": references})
            df.to_csv(os.path.join(self.log_dir, "validation_prediction-"+str(self.current_epoch)+".csv"))


