#!/bin/zsh 
eval "$(conda shell.bash hook)"
conda activate py36
#*wadi用的这个代码，但是也是改果德
# python data_preprocess.py WADI
#*这个用的师兄的，并且加了新东西
# 另一个处理结构
python processtest.py WADI
#/home/yiruolei/ALLDATASET/anomaly/data/WADI
