import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.model.mbart_qg import MBARTQG, MBARTQGDataLoaderCollator
from src.eval_utils.evaluate_utils import HFMetric, MultiHFMetric

import spacy
os.environ["TOKENIZERS_PARALLELISM"] = "false"
class SpacyTokenizer():
    def __init__(self):
        self.nlp = spacy.load("fr_core_news_lg")
    def __call__(self, x):
        return [t.text for t in self.nlp.tokenizer(x)]
st = SpacyTokenizer()
validation_metrics = MultiHFMetric(
    sacrebleu = HFMetric('sacrebleu', lambda x : x['score'], tokenize = 'intl'),
    rouge = HFMetric('rouge', lambda x : x['rougeL'], tokenizer = st)
)

os.environ['EFQADATA'] = '/people/gerald/Documents/repositories/Educational-French-Question-Answering/dataset'
data_folder = os.path.expandvars("$EFQADATA/source")
train_datasets_name = ["squad-en-en.pb.json","fquad-fr-fr.pb.json"]
valid_datasets_name = ["fquad-fr-fr.pb.json"]
train_datasets = {}
valid_datasets = {}

for dataset_name in train_datasets_name: 
    with open(os.path.join(data_folder, dataset_name)) as f:
        il, ol = dataset_name.split('.')[0].split('-')[-2], dataset_name.split('.')[0].split('-')[-1]
        data = json.load(f)
        train_datasets[dataset_name.split('.')[0]] = FQAGPBDataset(
            data["train"],
            sampler = lambda x : [x[random.randint(0, len(x) - 1)]],
            input_lang = il, output_lang = ol
        )
for dataset_name in valid_datasets_name: 
    with open(os.path.join(data_folder, dataset_name)) as f:
        il, ol = dataset_name.split('.')[0].split('-')[-2], dataset_name.split('.')[0].split('-')[-1]
        data = json.load(f)
        valid_datasets[dataset_name.split('.')[0]] = FQAGPBDataset(
            data["valid"],
            sampler = lambda x : [x[random.randint(0, len(x) - 1)]],
            input_lang = il, output_lang = ol
        )

model = MBARTQG(
    pretrained_name = "facebook/mbart-large-50-many-to-many-mmt",
    fixed_encoder = True,
    validation_callback = validation_metrics
)

tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.expandvars("$QA_LOG"), name="test")
#tb_logger.log_hyperparams(vars(args))
lr_monitor = LearningRateMonitor(logging_interval='step')

train_dl  = DataLoader(KeyMapDataset(MixedDataset(*train_datasets.values())), batch_size = 2, shuffle=True, num_workers=2, collate_fn=MBARTQGDataLoaderCollator(model.tokenizer))
valid_dl  = DataLoader(KeyMapDataset(MixedDataset(*valid_datasets.values())), batch_size = 2, shuffle=False, num_workers=2, collate_fn=MBARTQGDataLoaderCollator(model.tokenizer))


checkpoint_callback_val_loss = ModelCheckpoint(monitor='val/loss', save_top_k=2, mode="min", filename="val-loss-checkpoint-{epoch:02d}-{val_loss:.2f}")
checkpoint_callback_val_sacrebleu = ModelCheckpoint(monitor='val/sacrebleu', save_top_k=2, mode="max", filename="val-sacrebleu-checkpoint-{epoch:02d}-{val_loss:.2f}")
checkpoint_callback_val_rouge = ModelCheckpoint(monitor='val/rouge', save_top_k=2, mode="max", filename="val-rouge-checkpoint-{epoch:02d}-{val_loss:.2f}")

callbacks = [
    lr_monitor,
    checkpoint_callback_val_loss,
    checkpoint_callback_val_rouge,
    checkpoint_callback_val_sacrebleu
]


trainer = pl.Trainer(
    logger=tb_logger, 
    log_every_n_steps=100, 
    callbacks=callbacks, 
    enable_progress_bar=True,
    limit_train_batches=10000, 
    max_epochs=250, 
    accumulate_grad_batches=64,
    accelerator='cpu'
)
trainer.fit(
    model,
    train_dl,
    valid_dl
)