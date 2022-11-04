from torch import optim 
OPTIM_MAP = {
    "adamw" : optim.AdamW,
    "sgd" : optim.SGD
}