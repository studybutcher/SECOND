#!/bin/zsh    
# eval "$(conda activate loranomaly  hook)"
export Name='可视化'
#*目前最优的就是这个'attnmamba+blockize'
export MODE='visual'
 export LOAD='/home/yiruolei/project/anomaly-detection/SecondTask/checkpoint/0-casestudy/WADI/best.pt'
export CUDA_VISIBLE_DEVICES=0
export data='WADI'

 python main.py --dataset $data --name $Name  --trainmode $MODE --loadpath $LOAD 

 

