import json 
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything

from src.model.barthez_qg import DataCollator, BarthezQA
# from src.data_utils.pb_corpus import FQAGPBDataset
# from src.data_utils.corpus import MixedDataset, KeyMapDataset
from src.eval_utils.evaluate_utils import HFMetric, MultiHFMetric

from datasets import load_dataset

from tqdm import tqdm
import spacy

import argparse

parser = argparse.ArgumentParser(
    description='Train Barthez model for question generation on french QA dataset'
    )
parser.add_argument('--cpu-only', dest="cpu_only", default=False, action='store_true',
                    help='do not use GPUs (for dev only)')
parser.add_argument('--ndevices', dest='ndevices', type=int, default=1)

parser.add_argument('--enable-progress-bar', dest="enable_progress_bar", default=True, action='store_true',
                    help='show progress bar' )

parser.add_argument('--name', dest="name", default="test-barthez")
parser.add_argument('--datasets-path', metavar='datasets_path',
                    default="src/data/")

parser.add_argument('--log-every-n-steps', dest="log_every_n_steps", default=128, type=int,
                    help='log frequency')
parser.add_argument('--batch-size', dest="batch_size", default=16, type=int)
parser.add_argument('--max-epochs', dest="max_epochs", default=50, type=int,
                    help='number of training epoch' )

parser.add_argument('--limit-train-batches', dest='limit_train_batches', default=800, type=int)
parser.add_argument('--limit-val-batches', dest='limit_val_batches', default=150, type=int)

args = parser.parse_args()


class SpacyTokenizer:
    # A simple tokenizer class for ROUGE evaluation
    def __init__(self):
        self.nlp = spacy.load("fr_core_news_lg")
    def __call__(self, x):
        return [t.text for t in self.nlp.tokenizer(x)]

def sb_score(x):
    return x['score']

def rouge_score(x):
    return x['rougeL']


def main():

    ### FROM THOMAS
    seed_everything(42, workers=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Loading the metrics
    st = SpacyTokenizer()
    # Use of the different HuggingFace metrics here sacrebleu and rouge
    validation_metrics = MultiHFMetric(
        sacrebleu = HFMetric('sacrebleu', sb_score, tokenize = 'intl'), # we define the fct sb_score and rouge_score above to avoid lambda fct
        rouge = HFMetric('rouge', rouge_score, tokenizer = st)
    )
    ### END FROM THOMAS

    # To store the logs
    log_folder = os.path.expandvars("/people/tamames/QA/logs")

    #Loading the model
    model = BarthezQA(#load_pretraned_model='/people/tamames/project/saved_models/barthez-e5',
        validation_callback = validation_metrics
        ) # ajouter ou log

    # Loading the datasets
    path = args.datasets_path
    files = {'train': [ path +'piaf/train.jsonl', path + 'fquad/train.jsonl'],
            'valid': [path + 'piaf/valid.jsonl', path + 'fquad/valid.jsonl']} 


    dataset = load_dataset("json", data_files = files)

    # Training and validation dataloader
    ## TODO: sampler dans le dataset pour faire des epoch avec un nb de batch limité
    train_dataloader = DataLoader(dataset["train"],
                                batch_size=args.batch_size,
                                drop_last=False,
                                collate_fn = DataCollator(model.tokenizer),
                                shuffle=True,
                                num_workers=2
                                )

    valid_dataloader = DataLoader(dataset["valid"],
                                batch_size=args.batch_size,
                                drop_last=False,
                                collate_fn = DataCollator(model.tokenizer),
                                shuffle=True,
                                num_workers=2
                                )
    
    ### FROM THOMAS
    # init the logger with the default tensorboard logger from lightning
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_folder, name=args.name) 
    tb_logger.log_hyperparams(vars(args))
    # We also log the learning rate, at each step
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # instanciate the differente callback for saving the model according to the different metrics
    checkpoint_callback_val_loss = ModelCheckpoint(monitor='val_loss', save_top_k=2, mode="min", filename="val-loss-checkpoint-{epoch:02d}-{val_loss:.2f}")
    checkpoint_callback_val_sacrebleu = ModelCheckpoint(monitor='val_sacrebleu', save_top_k=2, mode="max", filename="val-sacrebleu-checkpoint-{epoch:02d}-{val_sacrebleu:.2f}")
    checkpoint_callback_val_rouge = ModelCheckpoint(monitor='val_rouge', save_top_k=2, mode="max", filename="val-rouge-checkpoint-{epoch:02d}-{val_rouge:.2f}")

    callbacks = [
        lr_monitor,
        checkpoint_callback_val_loss,
        checkpoint_callback_val_rouge,
        checkpoint_callback_val_sacrebleu
    ]
    ### END FROM THOMAS
    
    # Instanciate the trainer
    trainer = Trainer(
        logger=tb_logger, 
        log_every_n_steps=args.log_every_n_steps, 
        callbacks=callbacks, 
        enable_progress_bar=args.enable_progress_bar,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        max_epochs=args.max_epochs, 
        deterministic=True,
        accumulate_grad_batches=8,
        accelerator='gpu' if(not args.cpu_only) else 'cpu',
        devices=args.ndevices,
        auto_select_gpus=True,
        strategy="ddp" # strategy to train the model on different machine
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader
    )
    
    
if __name__ == "__main__":
    main()
    
    