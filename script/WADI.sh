#!/bin/zsh 
eval "$(conda activate loranomaly  hook)"
# conda 

export Name='casestudy补充'
# export Name='mambatest'
export data='WADI'
export MODE='train-test'
# export MODE='test'
export LOAD=None
# export LOAD='/home/yiruolei/project/anomaly-detection/SecondTask/checkpoint/mambatest/WADI/best.pt'

nohup python main.py --dataset $data --name $Name  --trainmode $MODE --loadpath $LOAD \
>> nohp-out/$Name.out 2>&1 &
