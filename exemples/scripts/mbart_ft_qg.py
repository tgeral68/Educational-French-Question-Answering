import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.model.mbart_qg import MBARTQG
from src.eval_utils.evalutation_utils import HFMetric, MultiHFMetric




def main():
    class SpacyTokenizer():
        def __init__(self):
            self.nlp = spacy.load("fr_core_news_lg")
        def __call__(self, text):
            return [t.text for t in self.nlp.tokenizer(x)]
    st = SpacyTokenizer()
    validation_metrics = MultiHFMetric(
        sacrebleu = HFMetric('sacrebleu', lambda x : x['score'], tokenize = 'intl'),
        rouge = HFMetric('rouge', lambda x : x['rougeL'], tokenizer = st)
    )

    os.environ['EFQADATA'] = '/people/gerald/Documents/repositories/Educational-French-Question-Answering/dataset'
    data_folder = os.path.expandvars("$EFQADATA/source")
    train_datasets_name = ["squad.pb.json","fquad.pb.json"]
    valid_datasets_name = ["fquad.pb.json"]
    train_datasets = {}
    valid_datasets = {}

    for dataset_name in train_datasets_name: 
        with open(os.path.join(data_folder, dataset_name)) as f:
            il, ol = dataset_name.split('-')[-2], dataset_name.split('-')[-1]
            data = json.load(f)
            train_datasets[dataset_name.split('.')[0]] = FQAGPBDataset(
                data["train"],
                sampler = lambda x : [x[random.randint(0, len(x) - 1)]],
                input_lang = il, output_lang = ol
            )
    for dataset_name in valid_datasets_name: 
        with open(os.path.join(data_folder, dataset_name)) as f:
            il, ol = dataset_name.split('-')[-2], dataset_name.split('-')[-1]
            data = json.load(f)
            valid_datasets[dataset_name.split('.')[0]] = FQAGPBDataset(
                data["valid"],
                sampler = lambda x : [x[random.randint(0, len(x) - 1)]],
                input_lang = il, output_lang = ol
            )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.expandvars("$QA_LOG"), name=args.name)
    tb_logger.log_hyperparams(vars(args))
    lr_monitor = LearningRateMonitor(logging_interval='step')

    train_dl  = DataLoader(KeyMapDataset(MixedDataset(*train_datasets.values())), batch_size = 2, shuffle=True, num_workers=2)
    valid_dl  = DataLoader(KeyMapDataset(MixedDataset(*valid_datasets.values())), batch_size = 2, shuffle=False, num_workers=2)


    checkpoint_callback_val_loss = ModelCheckpoint(monitor='val/loss', save_top_k=2, mode="min", filename="val-loss-checkpoint-{epoch:02d}-{val_loss:.2f}")
    checkpoint_callback_val_sacrebleu = ModelCheckpoint(monitor='val/sacrebleu', save_top_k=2, mode="min", filename="val-loss-checkpoint-{epoch:02d}-{val_loss:.2f}")
    checkpoint_callback_val_rouge = ModelCheckpoint(monitor='val/rouge', save_top_k=2, mode="min", filename="val-loss-checkpoint-{epoch:02d}-{val_loss:.2f}")

    callbacks = [lr_monitor, checkpoint_callback_val_loss, checkpoint_callback_val_rouge, checkpoint_callback_val_sacrebleu]

    print(64/ngpu, 'gpu' if(not args.cpu_only) else 'cpu')
    trainer = pl.Trainer(
        logger=tb_logger, 
        log_every_n_steps=100, 
        callbacks=callbacks, 
        enable_progress_bar=args.enable_progress_bar,
        resume_from_checkpoint=args.resume_from_checkpoint,  
        limit_train_batches=10000, 
        max_epochs=250, 
        accumulate_grad_batches=64//ngpu, 
        accelerator='gpu' if(not args.cpu_only) else 'cpu'
    )
    trainer.fit(MBARTQG(
            pretrained_name = "facebook/mbart-large-50-many-to-many-mmt",
            fixed_encoder = False,
            validation_callback = validation_metrics),
             train_dl, valid_dl)


if __name__ == "__main__":
    main()

# import os

# import torch
# import torch.utils.data as data

# import pytorch_lightning as pl
# from pytorch_lightning import loggers as pl_loggers
# from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# from transformers import AutoTokenizer

# from datasets import load_dataset
# from torch.utils.data import DataLoader



# from ..corpus.multimodal_corpus import Decorate, FQASDataset, FQGDataset, PQGDataset, MixedDataset, MultiModalDataColator, SubsetDataset
# from ..model.MT5Lightning import MT5

# import argparse

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--passage-selection-loss', dest="psl", type=float, default=0.,
#                     help='Weight scalar associated to the passage selection loss (on encoder part)')
# parser.add_argument('--cpu-only', dest="cpu_only", default=False, action='store_true',
#                     help='do not use GPUs (for dev only)')
# parser.add_argument('--enable-progress-bar', dest="enable_progress_bar", default=False, action='store_true',
#                     help='show progress bar' )
# parser.add_argument('--name', dest="name", default="default",
#                     help='set the name of the experiment' )
# parser.add_argument('--resume-from-checkpoint', dest="resume_from_checkpoint", type=str,  default=None,
#                     help='path if resuming training from checkpoint' )

# args = parser.parse_args()




# def build_dataset(split="train"):
#     data_files = {"train": "train.jsonl", "validation": "validation.jsonl"}
#     piaf_and_fquad = load_dataset(os.path.expandvars("$QA_DATASET/piaf_and_fquad/fquad_piaf_with_en"), data_files=data_files)
#     squad = load_dataset(os.path.expandvars("$QA_DATASET/squad/squad_with_fr"), data_files=data_files)
#     msmarco = load_dataset(os.path.expandvars("$QA_DATASET/msmarco/msmarco_generation_with_fr_final"), data_files=data_files)
#     sciq = load_dataset(os.path.expandvars("$QA_DATASET/SciQ"), data_files=data_files)
#     # Create two different datasets one for Question Answering selection
#     # piaf and fquad
#     qa_piaf_fquad = MixedDataset(*[
#         Decorate(FQGDataset(piaf_and_fquad[split]), prefix_list=[('tgt_txt', '[lang_fr]')]),
#         Decorate(FQGDataset(piaf_and_fquad[split], question_key="en_question"), prefix_list=[('tgt_txt', '[lang_en]')])
#     ])

#     # squad
#     qa_squad = MixedDataset(*[
#         Decorate(FQGDataset(squad[split]), prefix_list=[('tgt_txt', '[lang_en]')]),
#         Decorate(FQGDataset(squad[split], question_key="fr_question"), prefix_list=[('tgt_txt', '[lang_fr]')])
#     ])

#     qa_dataset = Decorate(
#         MixedDataset(
#             qa_piaf_fquad,
#             qa_squad,
#         )
#     )
#     return qa_dataset

# def main():
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
#     pl.utilities.seed.seed_everything(42)
#     print("BUILDING TRAINING SET", flush=True)
#     train_qa_dataset = build_dataset()
#     print("BUILDING VALIDATION SET", flush=True)
#     validation_qa_dataset = SubsetDataset(build_dataset("validation"), size=2048)
#     print("TRAIN SIZE %s \n VALID SIZE %s"%(len(train_qa_dataset), len(validation_qa_dataset)))
#     ngpu = torch.cuda.device_count() if(not args.cpu_only) else 1

#     tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.expandvars("$QA_LOG"), name=args.name)
#     tb_logger.log_hyperparams(vars(args))
#     lr_monitor = LearningRateMonitor(logging_interval='step')
#     tokenizer = AutoTokenizer.from_pretrained("google/mt5-base") #, use_fast=False)
#     tokenizer.add_tokens(['[question_generation]', '[highlight]', '[answer]','[ranking_query]', '[ranking_answer]', '[answer_generation]', '[answer_selection]', '[question]', '[lang_fr]', '[lang_en]'])

#     tdl = DataLoader(train_qa_dataset, shuffle=True, batch_size=2, collate_fn=MultiModalDataColator(tokenizer), num_workers=2)
#     vdl = DataLoader(validation_qa_dataset, shuffle=False, batch_size=2, collate_fn=MultiModalDataColator(tokenizer), num_workers=2)
#     checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=2, mode="min", filename="val-loss-checkpoint-{epoch:02d}-{val_loss:.2f}")
#     print(64/ngpu, 'gpu' if(not args.cpu_only) else 'cpu')
#     trainer = pl.Trainer(logger=tb_logger, log_every_n_steps=100, callbacks=[lr_monitor, checkpoint_callback], enable_progress_bar=args.enable_progress_bar,
#         resume_from_checkpoint=args.resume_from_checkpoint,  
#         limit_train_batches=10000, max_epochs=250,  accumulate_grad_batches=64//ngpu, accelerator='gpu' if(not args.cpu_only) else 'cpu')
#     trainer.fit(MT5(tokenizer, passage_selection_loss=args.psl), tdl, vdl)

# if __name__ == "__main__":
#     main()
