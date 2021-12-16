import yaml
import os

class configs:
    def __init__(self):
        self.TRAIN_DATASET_PATH = ""
        self.TRAIN_DATASET_LABEL_PATH = ""
        self.VAL_DATASET_PATH = ""
        self.VAL_DATASET_LABEL_PATH = ""
        self.MODEL_PATH = ""

        self.PRE_TRAIN = False
        self.TRAIN_BATCH_SIZE = 0
        self.TRAIN_NUM_WORKS = 0
        self.VAL_BATCH_SIZE = 0
        self.VAL_NUM_WORKS = 0

        self.NUM_EPOCHS = 0
        self.LAST_EPOCHS = 0
        self.WEIGHT_DECAY = 0
        self.BASE_LR = 0
        self.USE_WARMUP = True
        self.WARMUP_EPOCHS = 0
        self.WARMUP_START_LR = 0
        self.END_LR = 0

def update_config_from_file(config, cfg_file):
    infile =  open(cfg_file, 'r')
    yaml_cfg = yaml.load(infile, Loader=yaml.FullLoader)

    DATA_PATH = yaml_cfg['DATA_PATH']
    config.TRAIN_DATASET_PATH = DATA_PATH['TRAIN_DATASET_PATH']
    config.TRAIN_DATASET_LABEL_PATH = DATA_PATH['TRAIN_DATASET_LABEL_PATH']
    config.VAL_DATASET_PATH = DATA_PATH['VAL_DATASET_PATH']
    config.VAL_DATASET_LABEL_PATH = DATA_PATH['VAL_DATASET_LABEL_PATH']
    
    

    TRAIN = yaml_cfg['TRAIN']
    config.PRE_TRAIN = TRAIN['PRE_TRAIN']
    if config.PRE_TRAIN:
        config.MODEL_PATH = yaml_cfg['MODEL_PATH']
        
    config.TRAIN_BATCH_SIZE = TRAIN['TRAIN_BATCH_SIZE']
    config.TRAIN_NUM_WORKS = TRAIN['TRAIN_NUM_WORKS']
    config.VAL_BATCH_SIZE = TRAIN['VAL_BATCH_SIZE']
    config.VAL_NUM_WORKS = TRAIN['VAL_NUM_WORKS']

    SCHEDULER = yaml_cfg['SCHEDULER']
    config.NUM_EPOCHS = SCHEDULER['NUM_EPOCHS']
    config.LAST_EPOCHS = SCHEDULER['LAST_EPOCHS']
    config.WEIGHT_DECAY = SCHEDULER['WEIGHT_DECAY']
    config.BASE_LR = SCHEDULER['BASE_LR']
    config.USE_WARMUP = SCHEDULER['USE_WARMUP']
    if config.USE_WARMUP:
        config.WARMUP_EPOCHS = SCHEDULER['WARMUP_EPOCHS']
        config.WARMUP_START_LR = SCHEDULER['WARMUP_START_LR']
        config.END_LR = SCHEDULER['END_LR']

def get_config(cfg_file=None):
    config = configs()
    if cfg_file:
        update_config_from_file(config, cfg_file)
    return config
