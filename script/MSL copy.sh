#!/bin/zsh 
eval "$(conda activate loranomaly  hook)"
# conda 

export Name='重复测试'
export data='MSL'
export MODE='train-test'
export LOAD=None
export CUDA_VISIBLE_DEVICES=0

for i in {1..10}
    do nohup python main.py --dataset $data --name $Name  --trainmode $MODE --loadpath $LOAD \
            >> nohp-out/$Name.out 2>&1 &
    pid=$!
    wait $pid
done


