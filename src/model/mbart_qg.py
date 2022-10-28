import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.nn import BCELoss

import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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
            load_lang=["fr", "en"]
        ):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_name)
        self.tokenizers = {lang: AutoTokenizer.from_pretrained(pretrained_name, src_lang=LANGUAGE_MAP[lang])
                           for lang in load_lang}
        tokenizer.add_tokens(['<hl>'], special_tokens=True) for tokenizer in self.tokenizers.values 
        self.model.resize_token_embeddings(len(next(iter(self.tokenizers.values()))))

    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # print("DEVICE", batch_idx, self.device, batch["index"], flush=True)
        output =\
            self.model(
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                decoder_input_ids = batch['decoder_input_ids'],
                labels = batch['labels']
            )
        
        selection_loss = batch['input_ids'].new(1).zero_()[0].float()
        if(self.psl > 0):
            if(len(batch["selection_vector"][batch["selection_vector"] == -100]) != batch["selection_vector"].shape.numel()):
                pred = torch.sigmoid(self.linear_classifier(output.encoder_last_hidden_state).squeeze(-1))

                pred = pred[batch["selection_vector"] != -100]
                sel_gt = batch["selection_vector"][batch["selection_vector"] != -100] 
            
                selection_loss = self.bce_loss(pred, sel_gt)

        loss = (1-self.psl) * output.loss + (self.psl) * selection_loss
        self.log("training_loss",  loss.item(), reduce_fx="mean")
        self.log("training_reconstruction_loss",  output.loss.item(), reduce_fx="mean")
        self.log("training_selection_loss", selection_loss.item(), reduce_fx="mean")
        return loss
    
    def predict_selection(self, batch):

        data = self.model.encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-4)
        scheduler = {
            "scheduler": LinearLR(optimizer, total_iters=1000, start_factor=1.0 / 1000.),
            "interval": "step",
            'name': 'lr_scheduler',
            "frequency": 1
        }
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        output =\
            self.model(
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                decoder_input_ids = batch['decoder_input_ids'],
                labels = batch['labels']
            )
        
        selection_loss = batch['input_ids'].new(1).zero_()[0].float()
        if(self.psl > 0):
           if(len(batch["selection_vector"][batch["selection_vector"] == -100]) != batch["selection_vector"].shape.numel()):
                pred = torch.sigmoid(self.linear_classifier(output.encoder_last_hidden_state).squeeze(-1))

                pred = pred[batch["selection_vector"] != -100]
                sel_gt = batch["selection_vector"][batch["selection_vector"] != -100] 
            
                selection_loss = self.bce_loss(pred, sel_gt)

        loss = (1-self.psl) * output.loss + (self.psl) * selection_loss
        self.log("val_loss",  loss.item(), reduce_fx="mean", sync_dist=True)
        self.log("val_reconstruction_loss",  output.loss.item(), reduce_fx="mean", sync_dist=True)
        self.log("val_selection_loss", selection_loss.item(), reduce_fx="mean", sync_dist=True)