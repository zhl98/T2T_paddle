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

from common import Attention as Attention_Pure 
from common import Unfold  
from common import add_parameter  
from common import DropPath, Identity, Mlp  
from common import orthogonal_, trunc_normal_, zeros_, ones_ 

from dataset import get_dataset
from t2t import t2t_vit_7




#use_gpu = True
#paddle.set_device('gpu:1') if use_gpu else paddle.set_device('cpu')


if __name__ == '__main__':
    

    model = t2t_vit_7(pretrained = True)
    model = paddle.Model(model)
    model.prepare(metrics=paddle.metric.Accuracy(topk=(1, 5)))  
    
    dataset_val = get_dataset( mode='val')
    #val_sampler = DistributedBatchSampler(dataset_val, batch_size=128, drop_last=False)
    #dataloader_val = paddle.io.DataLoader(dataset_val,batch_sampler = val_sampler,num_workers=12)

    acc = model.evaluate(dataset_val, batch_size=128, num_workers=4, verbose=1)

   
    
