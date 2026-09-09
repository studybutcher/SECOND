#!/bin/zsh    
# eval "$(conda activate loranomaly  hook)"
export MODE='train-test'
export LOAD=None 
export data='MSL'
export CUDA_VISIBLE_DEVICES=1
export Name="Robust/${data}"
# export noise=0.1
for noise in {0.2,0.5} #0.05,0.10,0.15,
    # nohup
    do 
    nohup python main.py --dataset $data \
                            --name $Name  \
                            --trainmode $MODE \
                            --loadpath $LOAD \
                            --noise $noise \
    >> nohp-out/${Name}.out 2>&1 &
    pid=$!
    wait $pid
done