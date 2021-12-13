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
from paddle.vision import transforms, datasets, image_load
from dataset import get_dataset
from t2t import t2t_vit_7




vals_transform = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


if __name__ == '__main__':

    data = image_load(r'lit_data\train\n07579787/n07579787_1228.JPEG').convert('RGB')
    data = vals_transform(data)
    data = data.unsqueeze(0)
    model = t2t_vit_7(pretrained = True)
    pre_y = model(data)
    
    print("class_id is: {}".format(pre_y.argmax().item()))
    #model = paddle.Model(model)
    #model.prepare(metrics=paddle.metric.Accuracy(topk=(1, 5)))  
    
    #dataset_val = get_dataset( mode='val')
    #val_sampler = DistributedBatchSampler(dataset_val, batch_size=128, drop_last=False)
    #dataloader_val = paddle.io.DataLoader(dataset_val,batch_sampler = val_sampler,num_workers=12)

    

   
    
