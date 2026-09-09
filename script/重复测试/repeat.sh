#!/bin/zsh 
eval "$(conda activate loranomaly  hook)"
# conda 

export Name='ablation/'
# export data='WADI'
export MODE='train-test'
export LOAD=None
export CUDA_VISIBLE_DEVICES=0
# export info='不使用对比学习？？'
#'SMAP','SWaT',
for data in {'MSL','SMAP','SWaT','WADI'}  #
    for i in {1..3}
        do nohup python main.py --dataset $data \
                            --name $Name  \
                            --trainmode $MODE \
                            --loadpath $LOAD \
        >> nohp-out/$Name/$data.out 2>&1 &
        pid=$!
        wait $pid
    # sleep 1
    done
done


