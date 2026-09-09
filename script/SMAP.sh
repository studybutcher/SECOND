#!/bin/zsh 
eval "$(conda shell.bash hook)"
conda activate loranomaly 

export Name='casestudy补充'
export CUDA_VISIBLE_DEVICES=1
export MODE='train-test'
export data='SMAP'
export LOAD=None
nohup python main.py --dataset $data --name $Name  --trainmode $MODE --loadpath $LOAD\
>> nohp-out/$Name.out 2>&1 &
#todo 目前是没有使用对比loss的，只是加了框架，这个loss加入后的效果明天是=