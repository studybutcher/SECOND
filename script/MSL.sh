#!/bin/zsh 
eval "$(conda activate loranomaly  hook)"
# conda 

# export Name='casestudy设置'
export data='MSL'
export MODE='train-test'
export LOAD=None
export CUDA_VISIBLE_DEVICES=0


# 'MSL',
# for data in {'SMAP','SWaT','WADI'}  #
#     for i in {1..10}
#         do nohup python main.py --dataset $data \
#                             --name $Name  \
#                             --trainmode $MODE \
#                             --loadpath $LOAD \
#         >> nohp-out/$Name/$data.out 2>&1 &
#         pid=$!
#         wait $pid
#     # sleep 1
#     done
# done
for i in {1..10}
    do nohup python main.py --dataset $data --name $Name  --trainmode $MODE --loadpath $LOAD \
            >> nohp-out/$Name.out 2>&1 &
    pid=$!
    wait $pid
done


