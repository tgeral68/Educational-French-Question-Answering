import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.nn import BCELoss

import pytorch_lightning as pl
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast



class MBARTQG(pl.LightningModule):
    LANGUAGE_MAP = {
        "fr" : "fr_XX",
        "en" : "en_XX",
        "es" : "es_XX",
        "ja" : "ja_XX",
        "pt" : "pt_XX",
        "tl" : "tl_XX"
    }

    def __init__(
            self,
            pretrained_name="facebook/mbart-large-50-many-to-many-mmt",
            fixed_encoder=False
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
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                decoder_input_ids = batch['decoder_input_ids'],
                labels = batch['labels']
            )


        #.model.shared.weight._grad[:-1] = 0
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
        self.log("training_reconstruction_loss",  output.loss.item(), reduce_fx="mean")

        # validation metrics based on generative approach
        generated_batch = self.model.generate(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'])
        generated_text = self.tokenizer.batch_decode(generated_batch, skip_special_tokens=True)
        ground_truth_text = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        scores = scores_callback(generated_text, ground_truth_text)

        self.log("val_loss",  loss.item(), reduce_fx="mean", sync_dist=True)
    
        self.log("val_reconstruction_loss",  output.loss.item(), reduce_fx="mean", sync_dist=True)
        self.log("val_selection_loss", selection_loss.item(), reduce_fx="mean", sync_dist=True)