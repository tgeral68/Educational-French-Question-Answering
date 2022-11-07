''' A script for prediction of mbart model.
    !!!!!Use only one GPU!!!!!
'''

import spacy
import json 
import os

from datasets import load_dataset

from torch.utils.data import DataLoader
import random
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything


from src.data_utils.pb_corpus import FQAGPBDataset
from src.data_utils.corpus import MixedDataset, KeyMapDataset, PrepositionalTokenDataset
from src.model.mbart_qg import MBARTQG, MBARTQGDataLoaderCollator
from src.eval_utils.evaluate_utils import HFMetric, MultiHFMetric


import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--cpu-only', dest="cpu_only", default=False, action='store_true',
                    help='do not use GPUs (for dev only)')
parser.add_argument('--enable-progress-bar', dest="enable_progress_bar", default=False, action='store_true',
                    help='show progress bar' )



parser.add_argument('--from-checkpoint', dest="from_checkpoint", type=str,  default=None,
                    help='path of the checkpoint used for prediction' )
parser.add_argument('--lr', dest="lr", type=float,  default=1e-4,
                    help='the learning rate value' )
parser.add_argument('--test-dataset-folder',  dest="test_dataset_folder", type=str, 
                    default="$EFQADATA/test_dataset",
                    help='the folder containing the testing sets')

parser.add_argument('--batch-size', dest='batch_size', type=int,
                    default=4,
                    help='the batch_size')
parser.add_argument('--use-task-token', dest='use_task_token', default=False, action='store_true',
                    help='do we use a special token for the encoder here [question_generation]')
args = parser.parse_args()

def main():

    seed_everything(42, workers=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


    model = MBARTQG.load_from_checkpoint(args.from_checkpoint)
    model.eval()
    # Raw data are located in the folder specified by EFQADATA env var in the folder source
    data_folder = os.path.expandvars(args.test_dataset_folder)
    datasets = []
    log_folder = "/" + "/".join(args.from_checkpoint.split("/")[:-1])+"/predictions"
    os.makedirs(log_folder, exist_ok=True)
    for dataset_filename in os.listdir(data_folder):
        dataset_name = dataset_filename.split(".")[0]
        print("START Prediction for dataset ", dataset_name, " at ", os.path.join(data_folder, dataset_filename))
        dataset =\
            KeyMapDataset(load_dataset('json', 
                          data_files=[os.path.join(data_folder, dataset_filename)], split="all")
                        )
        print("The dataset is loaded", flush=True)
        data_loader =\
            DataLoader(dataset,
                       batch_size=args.batch_size,
                       shuffle=False,
                       num_workers=4,
                       collate_fn=MBARTQGDataLoaderCollator(model.tokenizer)
                    )
        print("The dataloader is loaded", flush=True)
        print("Starting the inference", flush=True)
        trainer =  Trainer(accelerator="gpu", devices=1)
        predictions = trainer.predict(model, data_loader)
        print("Saving the prediction at", os.path.join(log_folder, 'prediction-'+dataset_name+'.csv'))
        pd.DataFrame(predictions).to_csv(os.path.join(log_folder, 'prediction-'+dataset_name+'.csv'), index=False)
        





if __name__ == "__main__":
    main()
