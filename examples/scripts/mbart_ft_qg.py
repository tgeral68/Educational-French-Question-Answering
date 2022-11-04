import spacy
import json 
import os

from torch.utils.data import DataLoader
import random
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything


from src.data_utils.pb_corpus import FQAGPBDataset
from src.data_utils.corpus import MixedDataset, KeyMapDataset
from src.model.mbart_qg import MBARTQG, MBARTQGDataLoaderCollator
from src.eval_utils.evaluate_utils import HFMetric, MultiHFMetric


import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--fixed-encoder', dest="fixed_encoder", default=False, action='store_true',
                    help='do the encoder part is fixed (initial weights)')
parser.add_argument('--cpu-only', dest="cpu_only", default=False, action='store_true',
                    help='do not use GPUs (for dev only)')
parser.add_argument('--enable-progress-bar', dest="enable_progress_bar", default=False, action='store_true',
                    help='show progress bar' )


parser.add_argument('--limit-train-batches', dest="limit_train_batches", default=20000, type=int,
                    help='Limit the number of batches for a trianing epoch' )
parser.add_argument('--log-every-n-steps', dest="log_every_n_steps", default=16, type=int,
                    help='log frequency' )


parser.add_argument('--name', dest="name", default="default",
                    help='set the name of the experiment' )
parser.add_argument('--resume-from-checkpoint', dest="resume_from_checkpoint", type=str,  default=None,
                    help='path if resuming training from checkpoint' )
parser.add_argument('--optimizer', dest="optimizer", type=str,  default="adamw",
                    help='the name of the optimizer [adamw, sgd]' )
parser.add_argument('--lr', dest="lr", type=float,  default=1e-4,
                    help='the learning rate value' )
parser.add_argument('--training-set', metavar='training_set', type=str, nargs='+',
                    default=["fquad-fr-fr.pb.json", "fquad-fr-en.pb.json",
                        "piaf-fr-en.pb.json", "piaf-fr-fr.pb.json",
                        "squad-en-en.pb.json", "squad-en-fr.pb.json"
                    ],
                    help='the name of the training set to use')
parser.add_argument('--validation-set', metavar='validation_set', type=str, nargs='+',
                    default=[
                        "fquad-fr-fr.pb.json", "piaf-fr-fr.pb.json"
                    ],
                    help='the name of the validation set to use')
args = parser.parse_args()

### UTILITIES FUNCTIONS DO NOT USE LAMBDA FUNCTION IN MULTIPROCESSING
def random_sampler(x):
    return [x[random.randint(0, len(x) - 1)]]
def first_sampler(x):
    return [x[0]]
def sacrebleu_select(x):
    return x['score']
def rouge_select(x):
    return x['rougeL']

class SpacyTokenizer:
    # A simple tokenizer class for ROUGE evaluation
    def __init__(self):
        self.nlp = spacy.load("fr_core_news_lg")
    def __call__(self, x):
        return [t.text for t in self.nlp.tokenizer(x)]

def main():

    seed_everything(42, workers=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Loading the metrics
    st = SpacyTokenizer()

    # Use of the different HuggingFace metrics here sacrebleu and rouge
    validation_metrics = MultiHFMetric(
        sacrebleu = HFMetric('sacrebleu', sacrebleu_select, tokenize = 'intl'),
        rouge = HFMetric('rouge', rouge_select, tokenizer = st)
    )


    # Raw data are located in the folder specified by EFQADATA env var in the folder source
    data_folder = os.path.expandvars("$EFQADATA/source")
    log_folder = os.path.expandvars("$EFQALOG")


    # Loading the training and validation sets
    train_datasets = {}
    valid_datasets = {}

    print("TRAINING SETS ", args.training_set, flush=True)
    for dataset_name in args.training_set: 
        with open(os.path.join(data_folder, dataset_name)) as f:
            il, ol = dataset_name.split('.')[0].split('-')[-2], dataset_name.split('.')[0].split('-')[-1]
            data = json.load(f)
            train_datasets[dataset_name.split('.')[0]] = FQAGPBDataset(
                data["train"],
                sampler = random_sampler,
                input_lang = il, output_lang = ol
            )
    print("TRAINING SETS ", args.validation_set, flush=True)
    for dataset_name in args.validation_set: 
        with open(os.path.join(data_folder, dataset_name)) as f:
            il, ol = dataset_name.split('.')[0].split('-')[-2], dataset_name.split('.')[0].split('-')[-1]
            data = json.load(f)
            valid_datasets[dataset_name.split('.')[0]] = FQAGPBDataset(
                data["valid"],
                sampler = first_sampler,
                input_lang = il, output_lang = ol
            )

    # Create the model
    model = MBARTQG(
        pretrained_name = "facebook/mbart-large-50-many-to-many-mmt", # the name of the pretrain model
        fixed_encoder = args.fixed_encoder, # Do we optimize the encoder if false finetuned all the model
        validation_callback = validation_metrics, # A validation metric callback must output a dictionary {metric_name_1: value_1, metric_name_2 value_2}
        log_dir = os.path.join(log_folder, args.name), # The log directory of the model it will save the validation output within it
        optimizer = args.optimizer,
        learning_rate = args.lr
    )

    # initialise the logger (using the default tensorboard logger from lightning)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_folder, name=args.name) 
    tb_logger.log_hyperparams(vars(args))
    # We also log the learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # instanciate the training and validation dataloader
    train_dl  = DataLoader(KeyMapDataset(MixedDataset(*train_datasets.values())), batch_size=4, shuffle=True, num_workers=4, collate_fn=MBARTQGDataLoaderCollator(model.tokenizer))
    valid_dl  = DataLoader(KeyMapDataset(MixedDataset(*valid_datasets.values())), batch_size=4, shuffle=False, num_workers=4, collate_fn=MBARTQGDataLoaderCollator(model.tokenizer))

    # instanciate the differente callback for saving the model according to the different metrics
    checkpoint_callback_val_loss = ModelCheckpoint(monitor='val/loss', save_top_k=1, mode="min", filename="val-loss-checkpoint-{epoch:02d}-{val/loss:.2f}")
    checkpoint_callback_val_sacrebleu = ModelCheckpoint(monitor='val/sacrebleu', save_top_k=1, mode="max", filename="val-sacrebleu-checkpoint-{epoch:02d}-{val/sacrebleu:.2f}")
    checkpoint_callback_val_rouge = ModelCheckpoint(monitor='val/rouge', save_top_k=1, mode="max", filename="val-rouge-checkpoint-{epoch:02d}-{val/rouge:.2f}")

    callbacks = [
        lr_monitor,
        checkpoint_callback_val_loss,
        checkpoint_callback_val_rouge,
        checkpoint_callback_val_sacrebleu
    ]
    # instanciate the trainer
    trainer = pl.Trainer(
        logger=tb_logger, 
        log_every_n_steps=args.log_every_n_steps, 
        callbacks=callbacks, 
        enable_progress_bar=args.enable_progress_bar,
        limit_train_batches=args.limit_train_batches, 
        max_epochs=250, 
        deterministic=True,
        accumulate_grad_batches=8,
        accelerator='gpu' if(not args.cpu_only) else 'cpu',
        devices=-1,
        auto_select_gpus=False
    )

    # start training
    trainer.fit(
        model,
        train_dl,
        valid_dl
    )

if __name__ == "__main__":
    main()
