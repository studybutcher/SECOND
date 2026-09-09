#!/bin/zsh    
# eval "$(conda activate loranomaly  hook)"
export Name='可视化--score测试'
#*目前最优的就是这个'attnmamba+blockize'
export MODE='visual'
export LOAD='/home/yiruolei/project/anomaly-detection/SecondTask/checkpoint/casestudy补充/SMAP/best.pt'
export CUDA_VISIBLE_DEVICES=1
#*singlestep test
export data='SMAP'

 python main.py --dataset $data --name $Name  --trainmode $MODE --loadpath $LOAD 