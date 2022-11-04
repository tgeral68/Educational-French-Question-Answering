import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.model.mbart_qg import MBARTQG, MBARTQGDataLoaderCollator
from src.eval_utils.evaluate_utils import HFMetric, MultiHFMetric

import spacy
import json 
import pandas as pd
import os

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--fixed-encoder', dest="fixed_encoder", default=False, action='store_true',
                    help='do the encoder part is fixed (initial weights)')
parser.add_argument('--cpu-only', dest="cpu_only", default=False, action='store_true',
                    help='do not use GPUs (for dev only)')
parser.add_argument('--enable-progress-bar', dest="enable_progress_bar", default=False, action='store_true',
                    help='show progress bar' )
parser.add_argument('--name', dest="name", default="default",
                    help='set the name of the experiment' )
parser.add_argument('--resume-from-checkpoint', dest="resume_from_checkpoint", type=str,  default=None,
                    help='path if resuming training from checkpoint' )
parser.add_argument('--training-set', metavar='training_set', type=str, nargs='+',
                    default=["fquad-fr-fr.pb.json", "fquad-fr-en.pb.json",
                        "piaf-fr-en.pb.json", "piaf-fr-fr.pb.json",
                        "squad-en-en.pb.json", "squad-en-fr.pb.json"
                    ]
                    help='the name of the training set to use')
parser.add_argument('--validation-set', metavar='validation_set', type=str, nargs='+',
                    default=[
                        "fquad-fr-fr.pb.json", "piaf-fr-fr.pb.json"
                    ]
                    help='the name of the validation set to use')

args = parser.parse_args()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Loading the metrics
    class SpacyTokenizer():
        # A simple tokenizer class for ROUGE evaluation
        def __init__(self):
            self.nlp = spacy.load("fr_core_news_lg")
        def __call__(self, x):
            return [t.text for t in self.nlp.tokenizer(x)]
    st = SpacyTokenizer()
    # Use of the different HuggingFace metrics here sacrebleu and rouge
    validation_metrics = MultiHFMetric(
        sacrebleu = HFMetric('sacrebleu', lambda x : x['score'], tokenize = 'intl'),
        rouge = HFMetric('rouge', lambda x : x['rougeL'], tokenizer = st)
    )

    # TODO : Change the definition of the env variables
    os.environ['EFQADATA'] = '/people/gerald/Documents/repositories/Educational-French-Question-Answering/dataset'

    # Raw data are located in the folder specified by EFQADATA env var in the folder source
    data_folder = os.path.expandvars("$EFQADATA/source")


    # Loading the training and validation sets
    train_datasets = {}
    valid_datasets = {}

    for dataset_name in args.training_set: 
        with open(os.path.join(data_folder, dataset_name)) as f:
            il, ol = dataset_name.split('.')[0].split('-')[-2], dataset_name.split('.')[0].split('-')[-1]
            data = json.load(f)
            train_datasets[dataset_name.split('.')[0]] = FQAGPBDataset(
                data["train"],
                sampler = lambda x : [x[random.randint(0, len(x) - 1)]],
                input_lang = il, output_lang = ol
            )
    for dataset_name in args.validation_set: 
        with open(os.path.join(data_folder, dataset_name)) as f:
            il, ol = dataset_name.split('.')[0].split('-')[-2], dataset_name.split('.')[0].split('-')[-1]
            data = json.load(f)
            valid_datasets[dataset_name.split('.')[0]] = FQAGPBDataset(
                data["valid"],
                sampler = lambda x : [x[0]],
                input_lang = il, output_lang = ol
            )

    # Create the model
    model = MBARTQG(
        pretrained_name = "facebook/mbart-large-50-many-to-many-mmt", # the name of the pretrain model
        fixed_encoder = args.fixed_encoder, # Do we optimize the encoder if false finetuned all the model
        validation_callback = validation_metrics, # A validation metric callback must output a dictionary {metric_name_1: value_1, metric_name_2 value_2}
        log_dir = os.path.join(os.path.expandvars("$QA_LOG"), args.name) # The log directory of the model it will save the validation output within it
    )

    # initialise the logger (using the default tensorboard logger from lightning)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.expandvars("$QA_LOG"), name=args.name) 
    tb_logger.log_hyperparams(vars(args))
    # We also log the learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # instanciate the training and validation dataloader
    train_dl  = DataLoader(KeyMapDataset(MixedDataset(*train_datasets.values())), batch_size = 2, shuffle=True, num_workers=2, collate_fn=MBARTQGDataLoaderCollator(model.tokenizer))
    valid_dl  = DataLoader(KeyMapDataset(MixedDataset(*valid_datasets.values())), batch_size = 2, shuffle=False, num_workers=2, collate_fn=MBARTQGDataLoaderCollator(model.tokenizer))

    # instanciate the differente callback for saving the model according to the different metrics
    checkpoint_callback_val_loss = ModelCheckpoint(monitor='val/loss', save_top_k=2, mode="min", filename="val-loss-checkpoint-{epoch:02d}-{val_loss:.2f}")
    checkpoint_callback_val_sacrebleu = ModelCheckpoint(monitor='val/sacrebleu', save_top_k=2, mode="max", filename="val-sacrebleu-checkpoint-{epoch:02d}-{val_loss:.2f}")
    checkpoint_callback_val_rouge = ModelCheckpoint(monitor='val/rouge', save_top_k=2, mode="max", filename="val-rouge-checkpoint-{epoch:02d}-{val_loss:.2f}")

    callbacks = [
        lr_monitor,
        checkpoint_callback_val_loss,
        checkpoint_callback_val_rouge,
        checkpoint_callback_val_sacrebleu
    ]

    # instanciate the trainer
    trainer = pl.Trainer(
        logger=tb_logger, 
        log_every_n_steps=16, 
        callbacks=callbacks, 
        enable_progress_bar=True,
        limit_train_batches=10000, 
        max_epochs=250, 
        accumulate_grad_batches=64,
        accelerator='cpu'
    )
    # start training
    trainer.fit(
        model,
        train_dl,
        valid_dl
    )