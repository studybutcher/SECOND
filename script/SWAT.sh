#!/bin/zsh    
eval "$(conda activate loranomaly  hook)"

export Name='casestudy补充'
export MODE='train-test'
# export MODE='test'
export LOAD=None
export data='SWaT'
export CUDA_VISIBLE_DEVICES=0
 python main.py --dataset $data --name $Name  --trainmode $MODE --loadpath $LOAD 