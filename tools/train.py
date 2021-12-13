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
dist.init_parallel_env()

if __name__ == '__main__':
    num_epochs = 200
    loss_fun  = nn.CrossEntropyLoss()

    model = t2t_vit_7(True)
    #model = paddle.Model(model)

    model = paddle.DataParallel(model)
    model.train()

    dataset_train = get_dataset(mode = 'train')
    #dataset_train = ImageFolder('ILSVRC2012_w/train', transforms_train)
    train_sampler = DistributedBatchSampler(dataset_train, batch_size=256, drop_last=False,shuffle=True )
    dataloader_train = paddle.io.DataLoader(dataset_train,num_workers=8,batch_sampler = train_sampler)

    #dataloader_train = paddle.io.DataLoader(dataset_train,batch_size=128,num_workers=48,shuffle=True)
    dataset_val = get_dataset( mode='val')
    val_sampler = DistributedBatchSampler(dataset_val, batch_size=64, drop_last=False)
    dataloader_val = paddle.io.DataLoader(dataset_val,batch_sampler = val_sampler,num_workers=4)

    scheduler = WarmupCosineScheduler(learning_rate = 0.0001,
                                          warmup_start_lr = 1e-8,
                                          start_lr=0.0001,
                                          end_lr = 5e-5,
                                          warmup_epochs = 5,
                                          total_epochs = 310,
                                          last_epoch = 0,
                                          )
                                
    optimizer = paddle.optimizer.AdamW(
                parameters=model.parameters(),
                learning_rate=scheduler,
                weight_decay=0.05,  #0.03可以试一试
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
                acc = paddle.metric.accuracy(input=pre_y, label=Y) 
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

    
    paddle.save(model.state_dict(), "./output/t2t_vit_7_200.pdparams")
    
    model.eval()
    val_accs = []
    for X,Y in dataloader_val:
        pre_y = model(X)      
        Y =paddle.reshape(Y,shape=[-1, 1]) 
        acc = paddle.metric.accuracy(input=pre_y, label=Y) 
        val_accs.append(acc)
    val_acc = np.sum(val_accs) / len(val_accs)
    print("==============================================\n") 
    print("==============================================\n") 
    print("ImageNet final val acc is:%.4f" %val_acc)    
   