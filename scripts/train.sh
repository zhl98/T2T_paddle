#在ai studio上需要进行数据集解压  具体可以看我公开的项目
#tar xvf /root/paddlejob/workspace/train_data/datasets/data79807/ILSVRC2012_w.tar

#训练命令  单机单卡
python  ./tools/train.py

#4块gpu进行训练-单机多卡
#python -m paddle.distributed.launch --gpus '0,1,2,3' ./tools/train.py
