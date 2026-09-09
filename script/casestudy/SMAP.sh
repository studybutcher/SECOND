#!/bin/zsh 
eval "$(conda activate loranomaly  hook)"
# conda 

# export Name='casestudy设置'
export data='MSL'
export LOAD=None
export CUDA_VISIBLE_DEVICES=0

python test.py --dataset $data --name $Name  --trainmode $MODE --loadpath $LOAD