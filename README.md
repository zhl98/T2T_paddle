# 基于Paddle实现  ——Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet
## 1. 简介
### 主要改进
* 在中型数据集（例如 ImageNet）上从头开始训练时，ViT 与CNN相比性能较差。作者发现这是因为：
    * （1）输入图像的简单标记化无法对相邻像素之间的重要局部结构（例如，边缘，线条）建模，从而导致其训练样本效率低；
    * （2）ViT的冗余注意力骨干网设计导致固定计算预算中有限的功能丰富性和有限的训练样本

为了克服这些限制，作者提出了一种新的 Tokens 到 Token 视觉 Transformer（T2T-ViT），逐层 Tokens 到 Token（T2T）转换，以通过递归聚集相邻对象逐步将图像结构化为 Tokens 变成一个 Token ，这样就可以对周围 Token 表示的局部结构进行建模，并可以减少 Token 长度。

* Tokens-to-Token（T2T）模块旨在克服ViT中简单Token化机制的局限性，它采用渐进式方式将图像结构化为 Token 并建模局部结构信息；
* 而 Tokens 的长度可以通过渐进式迭代降低，每个 T2T 过程包含两个步骤：Restructurization 与 SoftSplit，见下图。  
![模型示意图](./images/t2t.png)


### aistudio体验教程:https://aistudio.baidu.com/aistudio/clusterprojectdetail/3179641

## 2. 数据集和复现精度
数据集使用ImageNet 2012的训练数据集，有1000类，训练集图片有1281167张，验证集图片有50000张，大小为144GB  
aistudio上的地址为：https://aistudio.baidu.com/aistudio/datasetdetail/79807  

|  网络   | steps  |  opt  | image_size   | batch_size | dataset   | epoch  |params_size|
|  ----  | ----  | ----    |  ----       | ----          |----  | ----  |----  |
| t2t-vit  | 1252  | AdamW | 224x224    |1024        |ImageNet| 320 |16.45MB|

目标精度：71.7%
实现：71.56%
模型参数已经在output文件夹中存放     

也可以在百度网盘上下载，下载地址：百度网盘链接：https://pan.baidu.com/s/1A2az_B51ywsUbDCAFvXTvQ 
提取码：6ib9     

### 2.1 log信息说明
训练过程可以看项目中的log文件夹下的信息，由于aistudio上的脚本任务最多只能运行72个小时，把训练过程分成多个步骤进行训练，可以看见log中的信息，当然4个train-0.log并不是在同一个环境上跑的，
*  train-0-(1).log是在aistudio上4块Tesla V100，batch_size为256*4     lr:采用先上升，在下降。从0.0002-线性上升到0.0010，再依次下降0.0005
*  train-0-(2).log环境是2块2080ti  ,   batch_size为128*2
*  train-0-(3).log环境是2块TITAN*24G,batch_size为256*2  log中包含了多次训练过程， lr最后一次采用 0.000075
* trainer-0-(4).log是最后在一块2080ti上训练的过程，最后导出了最好的模型，batch_size为128，避免了多块卡上验证精度不同的问题。  lr也是逐步下降，最后为0.000005
* trainer-0-信息不全.log 是在一开始跑的，跑了250个epoch已经很接近结果了，但是因为aistudio只能运行72小时，然后模型也没保存，学习率等参数也没打印出来，lr为一直不变的0.00002，batch_size为256*4
* val-workerlog.0 是最后在一块卡上的验证结果，可以用来参考验收

## 3. 准备环境
* 硬件：Tesla V100 * 4
* 框架：PaddlePaddle == 2.2.0
* 本代码在AIstudio上可以通过fork立马运行，只需要执行里面的val.sh即可避免了环境配置的各种问题
## 4. 快速开始
### 第一步：克隆本项目
```
    #clone this repo    
    https://github.com/zhl98/T2T_paddle.git
    cd T2T_paddle
```
### 第二步：修改代码路径
修改dataset.py中的数据集路径    
1. 修改dataset的地址
2. 修改label.txt的地址    
项目中默认使用lit_data中的路径进行测试
### 第三步：训练模型
运行sh文件，在文件中可以选择单卡或是多卡训练  
```
    bash ./scripts/train.sh
```
部分训练日志如下所示。
```
Epoch [98/200], Step [300/1252], Loss: 1.4250,acc: 0.6624, read_time: 0.0069, train_time: 0.4234, lr: 0.0009
Epoch [98/200], Step [400/1252], Loss: 1.4264,acc: 0.6627, read_time: 0.0037, train_time: 0.3946, lr: 0.0009
```
### 第四步：验证模型
```
    bash ./scripts/val.sh
```
部分验证日志如下所示。
```
Step [180/196], acc: 0.7163, read_time: 1.4773
Step [190/196], acc: 0.7157, read_time: 1.1667
ImageNet final val acc is:0.7156
```
### 第五步：验证预测
```
    python ./tools/predict.py
```
![模型示意图](./images/n07579787_1228.JPEG)    

输出结果为

```
    class_id is: 923
```
对照lit_data中的标签，可知预测正确
## 5.代码结构



```

|-- T2T_ViT_Paddle
    |-- log      #日志
    |   |-- trainer-0-信息不全.log 
    |   |-- val-workerlog.0    #验证实验结果
    |   |-- trainer-0-(1).log   #有时间信息  第一步
    |   |-- trainer-0-(2).log   # 第二步训练
    |   |-- trainer-0-(3).log   # 第三步训练
    |   |-- trainer-0-(4).log   # 在单卡上训练模型
    |-- lit_data    #模型目录
    |-- output    #模型目录
    |-- scripts   #运行脚本
    |   |-- eval.sh
    |   |-- train.sh
    |-- tools   #源码文件
        |-- common.py    #基础类的封装
        |-- dataset.py	 #数据集的加载
        |-- scheduler.py #学习率的跟新
        |-- t2t.py		 #网络模型定义	
        |-- train.py	 #训练代码
        |-- val.py		 #验证代码
        |-- predict.py	 #预测代码
    |-- README.md      
    |-- requirements.txt

```



