import math
import numpy as np

import paddle
import paddle.nn as nn
import os
import cv2
import paddle.vision.transforms as T
import paddle.distributed as dist
from paddle.io import DistributedBatchSampler, DataLoader
from scheduler import WarmupCosineScheduler
from PIL import Image
from paddle.vision import transforms
import time 
from common import Attention as Attention_Pure 
from common import Unfold  
from common import add_parameter  
from common import DropPath, Identity, Mlp  
from common import orthogonal_, trunc_normal_, zeros_, ones_ 
from config import get_config,configs
from dataset import get_dataset
from t2t import t2t_vit_7
from sys import argv
import sys, getopt



#use_gpu = True
#paddle.set_device('gpu:1') if use_gpu else paddle.set_device('cpu')
dist.init_parallel_env()

if __name__ == '__main__':

    file_path = './config/t2t_vit_7.yaml'
    '''
    opts, args = getopt.getopt(sys.argv[1:], "t:")
    for op, value in opts:
        if op == "-t":
            file_path = value
            print(file_path)
    '''
    config = get_config(file_path)

    model = t2t_vit_7(pretrained = config.PRE_TRAIN,model_path = config.MODEL_PATH)
    model = paddle.Model(model)
    model.prepare(metrics=paddle.metric.Accuracy(topk=(1, 5)))  
    
    
    dataset_val = get_dataset( config.VAL_DATASET_PATH, config.VAL_DATASET_LABEL_PATH,mode = 'val' )
    #val_sampler = DistributedBatchSampler(dataset_val, batch_size=128, drop_last=False)
    #dataloader_val = paddle.io.DataLoader(dataset_val,batch_sampler = val_sampler,num_workers=12)

    acc = model.evaluate(dataset_val, batch_size=config.VAL_BATCH_SIZE, num_workers=config.VAL_NUM_WORKS, verbose=1)

   
    
