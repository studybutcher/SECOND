#!/bin/zsh    
# eval "$(conda activate loranomaly  hook)"
export time=$(date "+%m-%d-%H:%M:%S")
export MODE='train-test'
export data='MSL'
# export LOAD=None 
# export windowlength=70
# export batchsize=60
# export latent_size=256

export CUDA_VISIBLE_DEVICES=0
export Name="Hyperpara/max_train_size_$time"

for max_train_size in {0.9,0.8,0.6,0.4}
    do nohup python main.py --dataset $data \
                            --trainmode $MODE \
                            --name $Name  \
                            --max_train_size $max_train_size\
    >> nohp-out/${Name}.out 2>&1 &
    pid=$!
    wait $pid
done

                   #     --loadpath $LOAD \
                        #     --batch_size $batchsize\
                        #     --latent_size $latent_size\
                        #     --window_length $windowlength\