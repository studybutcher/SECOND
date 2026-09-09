#!/bin/zsh    
# eval "$(conda activate loranomaly  hook)"
export MODE='train-test'
export LOAD=None 
export data='MSL'
export windowlength=70
export batchsize=60
export batchsize=4
export CUDA_VISIBLE_DEVICES=1
export Name='Hyperpara/batchsize'
# for latent_size in {40,128,320,512}
export latent_size=256
    # nohup
    # do
     nohup python main.py --dataset $data \
                            --name $Name  \
                            --trainmode $MODE \
                            --loadpath $LOAD \
                            --batch_size $batchsize\
                            --latent_size $latent_size\
                            --window_length $windowlength\
    >> nohp-out/${Name}.out 2>&1 &
    pid=$!
    wait $pid
# done