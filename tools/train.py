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
    opts, args = getopt.getopt(sys.argv[1:], "t:")
    for op, value in opts:
        if op == "-t":
            file_path = value
            print(file_path)
    
    config = get_config(file_path)

    num_epochs = config.NUM_EPOCHS
    loss_fun  = nn.CrossEntropyLoss()
    if config.PRE_TRAIN:
        model = t2t_vit_7(pretrained = config.PRE_TRAIN,model_path = config.MODEL_PATH)
    else:
        model = t2t_vit_7(pretrained = config.PRE_TRAIN)
    #model = paddle.Model(model)

    model = paddle.DataParallel(model)
    model.train()

    dataset_train = get_dataset(config.TRAIN_DATASET_PATH,config.TRAIN_DATASET_LABEL_PATH,mode = 'train')
    #dataset_train = ImageFolder('ILSVRC2012_w/train', transforms_train)
    train_sampler = DistributedBatchSampler(dataset_train, batch_size = config.TRAIN_BATCH_SIZE, drop_last=False,shuffle=True )
    dataloader_train = paddle.io.DataLoader(dataset_train,num_workers = config.TRAIN_NUM_WORKS,batch_sampler = train_sampler)


    dataset_val = get_dataset( config.VAL_DATASET_PATH, config.VAL_DATASET_LABEL_PATH,mode = 'val' )
    val_sampler = DistributedBatchSampler(dataset_val, batch_size = config.VAL_BATCH_SIZE, drop_last=False)
    dataloader_val = paddle.io.DataLoader(dataset_val,batch_sampler = val_sampler,num_workers = config.VAL_NUM_WORKS)

    
    if config.USE_WARMUP:
        scheduler = WarmupCosineScheduler(learning_rate = config.BASE_LR,
                                          warmup_start_lr = float(config.WARMUP_START_LR),
                                          start_lr = config.BASE_LR,
                                          end_lr = float(config.END_LR),
                                          warmup_epochs = config.WARMUP_EPOCHS,
                                          total_epochs = config.NUM_EPOCHS,
                                          last_epoch = config.LAST_EPOCHS,
                                          )
        optimizer = paddle.optimizer.AdamW(
                    parameters = model.parameters(),
                    learning_rate = scheduler,
                    weight_decay = config.WEIGHT_DECAY,  #0.03可以试一试
                    beta1 = 0.9,
                    beta2 = 0.999,
                    epsilon = 1e-8,
                    grad_clip = paddle.nn.ClipGradByGlobalNorm(1.0) )
    else:
        optimizer = paddle.optimizer.AdamW(
                    parameters = model.parameters(),
                    learning_rate = config.BASE_LR,
                    weight_decay = config.WEIGHT_DECAY,  #0.03可以试一试
                    beta1 = 0.9,
                    beta2 = 0.999,
                    epsilon = 1e-8,
                    grad_clip = paddle.nn.ClipGradByGlobalNorm(1.0) )
    #optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=0.05,weight_decay = 0.03)

    #model.prepare(optimizer = optimizer,loss = loss_fun,metrics=paddle.metric.Accuracy(topk=(1, 5)))
    #model.fit(dataset_train,epochs = num_epochs,batch_size=128,verbose=1)
    
    total_step = len(dataloader_train)
    read_start_time = 0
    read_time = 0
    train_start_time = 0
    train_time = 0
    flag_lr = 0
    max_acc = 0.71
    for epoch in range(0,num_epochs):
        model.train()
        i = 0
        train_losses = []
        train_accs = []
        #read_time
        read_start_time = time.time()
        for X,Y in dataloader_train:
            read_time+=(time.time()-read_start_time)

            i = i + 1 
            #train time
            train_start_time = time.time()
            pre_y = model(X)
            loss = loss_fun(pre_y,Y)
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            train_time += (time.time()-train_start_time)

            train_losses.append(loss.item())       
            Y = paddle.reshape(Y,shape=[-1, 1]) 
            acc = paddle.metric.accuracy(input=pre_y, label=Y)
         
            train_accs.append(acc)
            if i%1000 == 0:
                train_loss = np.sum(train_losses) / len(train_losses)
                train_acc = np.sum(train_accs) / len(train_accs)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, read_time: {:.4f}, train_time: {:.4f}, lr: {:.6f}'.format(epoch, num_epochs, i, total_step, train_loss,read_time/1000,train_time/1000,optimizer.get_lr()))
                read_time=0
                train_time=0
            read_start_time = time.time()
        scheduler.step() 
        

        train_loss = np.sum(train_losses) / len(train_losses)
        train_acc = np.sum(train_accs) / len(train_accs)
        print('Epoch [{}/{}], avg_Loss: {:.4f}, avg_acc: {:.4f}'.format(epoch, num_epochs,  train_loss,train_acc))
        
        
        if train_acc > 0.70:
            model.eval()
            val_accs = []
            for X,Y in dataloader_val:
                pre_y = model(X)      

                Y =paddle.reshape(Y,shape=[-1, 1])  
                all_Y = []
                paddle.distributed.all_gather(all_Y, Y)
                all_labels = paddle.concat(all_Y, 0)
        
                all_pre_y = []
                paddle.distributed.all_gather(all_pre_y, pre_y)
                all_pre = paddle.concat(all_pre_y, 0)
        
                acc = paddle.metric.accuracy(input=all_pre, label=all_labels) 
                val_accs.append(acc)
            val_acc = np.sum(val_accs) / len(val_accs)
            print("ImageNet val acc is:%.4f" %val_acc)
            if val_acc > max_acc:
                max_acc = val_acc
                print('avg_acc: {:.4f} model saved!'.format( val_acc))
                paddle.save(model.state_dict(), "./output/t2t_vit_7_max.pdparams")
            if(val_acc > 0.7155):
                print("model saved!\n")  
                paddle.save(model.state_dict(), "./output/t2t_vit_7_final.pdparams")


    print("train ended!\n")  
 
    model.eval()
    val_accs = []
    for X,Y in dataloader_val:
        pre_y = model(X)      
        Y =paddle.reshape(Y,shape=[-1, 1])  
        all_Y = []
        paddle.distributed.all_gather(all_Y, Y)
        all_labels = paddle.concat(all_Y, 0)
        
        all_pre_y = []
        paddle.distributed.all_gather(all_pre_y, pre_y)
        all_pre = paddle.concat(all_pre_y, 0)
        
        acc = paddle.metric.accuracy(input=all_pre, label=all_labels) 
        val_accs.append(acc)
    val_acc = np.sum(val_accs) / len(val_accs)
    print("==============================================\n") 
    print("==============================================\n") 
    print("ImageNet final val acc is:%.4f" %val_acc)    
   