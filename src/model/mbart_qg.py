import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.nn import BCELoss

import pytorch_lightning as pl
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
LANGUAGE_MAP = {
    "fr" : "fr_XX",
    "en" : "en_XX",
    "es" : "es_XX",
    "ja" : "ja_XX",
    "pt" : "pt_XX",
    "tl" : "tl_XX"
}





class MBARTQG(pl.LightningModule):
    def prepare_input(self, batch: dict) -> dict:

        source = self.tokenizer(batch['context'], return_tensors="pt",  padding='longest', truncation=True, max_length=512)
        target = self.tokenizer(batch['question'], return_tensors="pt",  padding='longest', truncation=True, max_length=512)

        source_input_ids = source.input_ids
        target_input_ids = target.input_ids

        source_input_ids[:, 0] = batch["input_lang"]
        target_input_ids[:, 0] = batch["output_lang"]
        
        return {
            "input_ids": source_input_ids,
            "attention_mask": source.attention_mask,
            "labels": target_input_ids
        }

    def __init__(
            self,
            pretrained_name = "facebook/mbart-large-50-many-to-many-mmt",
            fixed_encoder = False,
            validation_callback = None
        ):
        super().__init__()
        self.fixed_encoder = fixed_encoder
        self.model = MBartForConditionalGeneration.from_pretrained(pretrained_name)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(pretrained_name)
        self.tokenizer.add_tokens(['<hl>'], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))


    
    def training_step(self, batch, batch_idx):
        output =\
            self.model(
                self.prepare_input(batch)
            )
        loss = output.loss 
        self.log("training_reconstruction_loss",  output.loss.item(), reduce_fx="mean")
        return loss
    
    def configure_optimizers(self):
        optimizable_parameters = list(self.model.decoder.parameters()) + list(self.model.lm_head.parameters()) 
        if(not self.fixed_encoder):
            optimizable_parameters = self.model.parameters()
        optimizer = AdamW(optimizable_parameters, lr=1e-4)
        scheduler = {
            "scheduler": LinearLR(optimizer, total_iters = 1000, start_factor= 1.0 / 1000.),
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
        generated_batch = self.model.generate(
            input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask'],
            forced_bos_token_id=tokenizer.lang_code_to_id[MBARTQG.LANGUAGE_MAP['fr']]
            )
        generated_text = self.tokenizer.batch_decode(generated_batch, skip_special_tokens=True)
        ground_truth_text = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        scores = scores_callback(generated_text, ground_truth_text)

        self.log("val/loss",  loss.item(), reduce_fx="mean", sync_dist=True)
        return {"generated_text": generated_text, "ground_truth_text":ground_truth_text}
    
    def validation_step_end(self, batch, outs):
        if self.validation_callback is not None:
            validation_log =  self.validation_callback(outs)
            for k, v in validation_log.items():
                self.log("val/"+k, v)

    def on_after_backward(self, trainer, pl_module):
        self.model.shared.weight._grad[:-1] = 0